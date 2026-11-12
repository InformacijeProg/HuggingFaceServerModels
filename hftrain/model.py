"""Loading the base model (through hfserve's registry), attaching LoRA adapters, and the
small model-related helpers shared by train/eval/export."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hfserve import engine, registry
from hfserve.registry import LocalModel

from . import config

log = logging.getLogger("hftrain.model")

# Opening reasoning tag at the end of a generation prompt, e.g. <ifm|think_fast>
_THINK_OPEN_RE = re.compile(r"(<(?!/)[^<>\s]*think[^<>\s]*>)\s*$", re.IGNORECASE)


def find_local(model_ref: str | None) -> LocalModel:
    local = registry.find_model(model_ref or config.DEFAULT_MODEL)
    if local.incomplete:
        raise registry.ModelError(f"{local.repo_id} looks like an interrupted download; run: hfserve pull {local.repo_id}")
    return local


def resolve_device_dtype(device: str | None, dtype: str | None) -> tuple[str, torch.dtype]:
    dev = device or engine.pick_device()
    dt = engine.parse_dtype(dtype) if dtype else engine.pick_dtype(dev)
    if dt == torch.bfloat16 and dev.startswith("cuda") and not engine._cuda_bf16_native():
        log.warning("This GPU has no native bfloat16; training will be slow. float16 is the default here.")
    return dev, dt


# ChatML-style assistant turn, as in EuroLLM's template. The patched form wraps the
# assistant content in {% generation %} markers so apply_chat_template can return an
# assistant-tokens mask and chat training puts loss on the answers only.
_CHATML_LOOP = """{% for message in messages %}{% if message['role'] != 'system' %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}"""
_CHATML_LOOP_MARKED = """{% for message in messages %}{% if message['role'] == 'assistant' %}<|im_start|>assistant
{% generation %}{{ message['content'] }}<|im_end|>
{% endgeneration %}{% else %}{% if message['role'] != 'system' %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endif %}{% endif %}{% endfor %}"""


def add_generation_markers(template: str | None) -> str | None:
    """The template with ``{% generation %}`` markers around assistant turns, or None when
    it already has them or is not the ChatML shape this patch knows."""
    if not template or "{% generation %}" in template or _CHATML_LOOP not in template:
        return None
    return template.replace(_CHATML_LOOP, _CHATML_LOOP_MARKED)


def load_tokenizer(local: LocalModel):
    tok = AutoTokenizer.from_pretrained(str(local.path), trust_remote_code=local.needs_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    patched = add_generation_markers(getattr(tok, "chat_template", None))
    if patched:
        tok.chat_template = patched
        log.info("chat template patched with {%% generation %%} markers (assistant-only loss)")
    return tok


def load_model(local: LocalModel, device: str, dtype: torch.dtype, *, training: bool = True):
    kwargs = {
        "trust_remote_code": local.needs_remote_code,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    if local.needs_remote_code:
        log.warning("%s ships custom model code (auto_map): loading with trust_remote_code=True", local.repo_id)
    model = AutoModelForCausalLM.from_pretrained(str(local.path), **kwargs)
    model.to(device)
    if training:
        model.config.use_cache = False
    return model


def add_lora(model, *, r: int, alpha: int, dropout: float, target_modules: tuple[str, ...] | list[str],
             train_embeddings: bool = False, gradient_checkpointing: bool = True):
    """Wrap ``model`` in a PEFT LoRA model. Adapter weights are kept in float32 by PEFT
    (needed for fp16 mixed precision); the base stays in its loaded dtype."""
    from peft import LoraConfig, get_peft_model

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    modules_to_save = ["embed_tokens", "lm_head"] if train_embeddings else None
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=list(target_modules), modules_to_save=modules_to_save,
    )
    return get_peft_model(model, lcfg)


def load_adapter(model, adapter_dir: Path, *, trainable: bool = False):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=trainable)


def param_summary(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def end_of_turn_ids(tokenizer) -> list[int]:
    """eos plus the token the chat template uses to close an assistant turn (some templates
    use a token that is not in generation_config; generation would run on without it)."""
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    try:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}], tokenize=True)
        if hasattr(enc, "keys"):
            enc = enc["input_ids"]
        last = int(enc[-1])
        added = set(tokenizer.get_added_vocab().values())
        if last not in ids and (last in added or last in set(tokenizer.all_special_ids)):
            ids.append(last)
    except Exception:  # noqa: BLE001 - no chat template or a template that rejects the probe
        pass
    return ids


def render_chat_prompt(tokenizer, text: str, *, reasoning_effort: str | None = "low", think: bool = False,
                       system: str | None = None) -> str:
    """User message -> prompt string with the generation prefix. With ``think=False`` an
    opening reasoning tag the template appends is closed immediately, so the model
    answers directly; that is what a quick language check wants."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except TypeError:  # template without reasoning_effort
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not think:
        m = _THINK_OPEN_RE.search(prompt)
        if m:
            opener = m.group(1)
            prompt = prompt[: m.end(1)] + "\n</" + opener[1:] + "\n"
    return prompt


def document_prefix_ids(tokenizer) -> list[int]:
    """Special tokens the tokenizer adds in front of a text by default (BOS ``<s>`` for
    Llama-style tokenizers such as EuroLLM's). Used at every document start when packing."""
    try:
        return list(tokenizer("", add_special_tokens=True)["input_ids"])
    except Exception:  # noqa: BLE001
        return []


def cuda_mem_mb() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    return int(torch.cuda.memory_allocated() // 2**20), int(torch.cuda.max_memory_allocated() // 2**20)
