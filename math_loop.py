"""Ask the local model random English math problems, one after another.

    python math_loop.py                 # until ESC (or Ctrl+C)
    python math_loop.py --count 20      # twenty problems, then stop
    python math_loop.py --model EuroLLM-1.7B-Instruct --temperature 0
    python math_loop.py --no-stream     # wait for the whole answer, then print it

The next problem goes out as soon as the previous answer is finished - there is no
pause in between unless ``--delay`` asks for one.

Because the problems are generated here, the right answer is known, so every reply is
also checked and the run ends with a score. The check is deliberately generous: any
number in the reply that equals the expected value counts as correct, since small
models like to answer in a sentence.

Press ESC (or q) at any time, even while an answer is streaming in, and the loop stops
at once; Ctrl+C does the same. The server is left as it was found - started here means
stopped here - and the model is unloaded from VRAM on the way out unless
``--keep-loaded`` says otherwise.

The address comes from hfserve.config, so HFSERVE_HOST / HFSERVE_PORT (or SERVER_URL)
in .env are what this script talks to; --host / --port override them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass

import httpx

from hfserve import config, process

SYSTEM_PROMPT = ("You are a careful mathematician. Solve the problem and reply with the final "
                 "numeric answer only - no words, no units, no working out.")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
ANSWER_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- problems
@dataclass
class Problem:
    question: str
    answer: float
    kind: str


def _int(value: float) -> str:
    """Render a number the way a person would: 42, not 42.0."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def p_add(rng: random.Random) -> Problem:
    a, b = rng.randint(120, 9999), rng.randint(120, 9999)
    return Problem(f"What is {a} + {b}?", a + b, "addition")


def p_subtract(rng: random.Random) -> Problem:
    a = rng.randint(500, 9999)
    b = rng.randint(100, a)  # keep it positive: easier to grade, no sign confusion
    return Problem(f"What is {a} - {b}?", a - b, "subtraction")


def p_multiply(rng: random.Random) -> Problem:
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    return Problem(f"What is {a} * {b}?", a * b, "multiplication")


def p_divide(rng: random.Random) -> Problem:
    divisor, quotient = rng.randint(3, 25), rng.randint(4, 99)
    return Problem(f"What is {divisor * quotient} / {divisor}?", quotient, "division")


def p_percent(rng: random.Random) -> Problem:
    percent = rng.choice([5, 10, 12, 15, 20, 25, 40, 50, 60, 75, 80])
    # The base has to be a multiple of 100/gcd(percent,100) for the answer to come out
    # whole - 12% of 760 is 91.2, which is a nastier thing to ask and to grade.
    step = 100 // math.gcd(percent, 100)
    base = rng.randint(2, max(2, 800 // step)) * step
    return Problem(f"What is {percent}% of {base}?", base * percent // 100, "percentage")


def p_power(rng: random.Random) -> Problem:
    base, exponent = rng.randint(2, 15), rng.randint(2, 3)
    return Problem(f"What is {base} to the power of {exponent}?", base ** exponent, "power")


def p_square_root(rng: random.Random) -> Problem:
    root = rng.randint(4, 30)
    return Problem(f"What is the square root of {root * root}?", root, "square root")


def p_average(rng: random.Random) -> Problem:
    count = rng.randint(3, 5)
    mean = rng.randint(10, 90)
    values = [rng.randint(1, 40) for _ in range(count - 1)]
    values.append(mean * count - sum(values))  # forces an exact integer mean
    rng.shuffle(values)
    listed = ", ".join(str(v) for v in values)
    return Problem(f"What is the average of {listed}?", mean, "average")


def p_equation(rng: random.Random) -> Problem:
    a, x, b = rng.randint(2, 12), rng.randint(2, 30), rng.randint(1, 50)
    return Problem(f"Solve for x: {a}x + {b} = {a * x + b}. What is x?", x, "linear equation")


def p_remainder(rng: random.Random) -> Problem:
    divisor = rng.randint(3, 19)
    dividend = rng.randint(2, 40) * divisor + rng.randint(1, divisor - 1)
    return Problem(f"What is the remainder when {dividend} is divided by {divisor}?",
                   dividend % divisor, "remainder")


def p_fraction_of(rng: random.Random) -> Problem:
    denominator = rng.choice([2, 3, 4, 5, 6, 8])
    numerator = rng.randint(1, denominator - 1)
    whole = denominator * rng.randint(3, 25)
    return Problem(f"What is {numerator}/{denominator} of {whole}?",
                   whole * numerator / denominator, "fraction")


def p_gcd(rng: random.Random) -> Problem:
    common = rng.randint(3, 15)
    a, b = common * rng.randint(2, 15), common * rng.randint(2, 15)
    return Problem(f"What is the greatest common divisor of {a} and {b}?",
                   math.gcd(a, b), "gcd")


def p_sequence(rng: random.Random) -> Problem:
    start, step = rng.randint(2, 30), rng.randint(2, 15)
    terms = [start + i * step for i in range(4)]
    listed = ", ".join(str(t) for t in terms)
    return Problem(f"What comes next in this sequence: {listed}, ?", terms[-1] + step, "sequence")


def p_shopping(rng: random.Random) -> Problem:
    price, quantity = rng.randint(3, 40), rng.randint(3, 12)
    paid = price * quantity + rng.randint(1, 50)
    return Problem(f"A notebook costs {price} euros. Someone buys {quantity} of them and pays "
                   f"with {paid} euros. How much change do they get?",
                   paid - price * quantity, "word problem")


def p_speed(rng: random.Random) -> Problem:
    speed, hours = rng.randint(20, 120), rng.randint(2, 9)
    return Problem(f"A train travels at {speed} km/h for {hours} hours. "
                   f"How many kilometres does it cover?", speed * hours, "word problem")


GENERATORS = (p_add, p_subtract, p_multiply, p_divide, p_percent, p_power, p_square_root,
              p_average, p_equation, p_remainder, p_fraction_of, p_gcd, p_sequence,
              p_shopping, p_speed)


def make_problem(rng: random.Random) -> Problem:
    return rng.choice(GENERATORS)(rng)


def check(reply: str, expected: float) -> tuple[bool, str | None]:
    """``(correct, the last number in the reply)``.

    Thousands separators are removed first, so "1,234" and "1 234" both read as 1234.
    """
    cleaned = re.sub(r"(?<=\d)[,  ](?=\d\d\d\b)", "", reply)
    found = NUMBER.findall(cleaned)
    if not found:
        return False, None
    values = []
    for text in found:
        try:
            values.append(float(text))
        except ValueError:
            continue
    correct = any(abs(v - expected) <= ANSWER_TOLERANCE for v in values)
    return correct, found[-1]


# --------------------------------------------------------------------------- ESC watcher
class StopKey:
    """Sets an event when ESC or q is pressed, without blocking anything."""

    KEYS = {b"\x1b": "ESC", b"q": "q", b"Q": "q"}

    def __init__(self) -> None:
        self.event = threading.Event()
        self.key = ""
        self.watching = False

    def start(self) -> None:
        target = self._watch_windows if os.name == "nt" else self._watch_posix
        try:
            threading.Thread(target=target, name="math-loop-keys", daemon=True).start()
            self.watching = True
        except RuntimeError:
            self.watching = False  # no thread: Ctrl+C still works

    def _hit(self, char: bytes) -> bool:
        name = self.KEYS.get(char)
        if name:
            self.key = name
            self.event.set()
            return True
        return False

    def _watch_windows(self) -> None:
        try:
            import msvcrt
        except ImportError:  # pragma: no cover - Windows only
            return
        while not self.event.is_set():
            try:
                if msvcrt.kbhit() and self._hit(msvcrt.getch()):
                    return
            except OSError:  # stdin is not a console (piped input): Ctrl+C only
                self.watching = False
                return
            time.sleep(0.05)

    def _watch_posix(self) -> None:
        try:
            import select
            import termios
            import tty
        except ImportError:  # pragma: no cover - POSIX only
            return
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
        except (ValueError, OSError, termios.error):
            return  # not a terminal: Ctrl+C only
        try:
            tty.setcbreak(fd)
            while not self.event.is_set():
                if select.select([fd], [], [], 0.05)[0] and self._hit(os.read(fd, 1)):
                    return
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            except (OSError, termios.error):
                pass


# --------------------------------------------------------------------------- the server
def pick_model(client: httpx.Client, wanted: str | None, health: dict | None) -> str:
    """The model to ask: the one requested, the one loaded, or the only one installed."""
    if wanted:
        return wanted
    loaded = str((health or {}).get("model") or "")
    if loaded:
        return loaded
    r = client.get("/api/tags", timeout=20)
    r.raise_for_status()
    names = [m.get("name", "") for m in (r.json() or {}).get("models") or []]
    names = [n for n in names if n]
    if len(names) == 1:
        return names[0]
    if not names:
        raise SystemExit("No models installed. Run:  hfserve pull <org/name>")
    listed = "\n  ".join(names)
    raise SystemExit(f"Several models are installed; pick one with --model:\n  {listed}")


def ask_streaming(client: httpx.Client, payload: dict, stop: threading.Event,
                  on_delta) -> tuple[str, dict]:
    """Stream one answer. Returns ``(text, final stats)``; stops early if *stop* is set."""
    text, stats = "", {}
    with client.stream("POST", "/api/chat", json={**payload, "stream": True}) as r:
        if r.status_code != 200:
            detail = r.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {r.status_code}: {detail[:300]}")
        for line in r.iter_lines():
            if stop.is_set():
                break  # leaving the block closes the connection; the server cancels
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            if chunk.get("done"):
                stats = chunk
                break
            delta = (chunk.get("message") or {}).get("content") or ""
            if delta:
                text += delta
                on_delta(delta)
    return text, stats


def ask_blocking(client: httpx.Client, payload: dict) -> tuple[str, dict]:
    r = client.post("/api/chat", json={**payload, "stream": False}, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json() or {}
    return ((data.get("message") or {}).get("content") or ""), data


def tokens_per_second(stats: dict) -> tuple[int, float]:
    """``(generated tokens, tokens per second)`` from the Ollama-style nanosecond stats."""
    count = int(stats.get("eval_count") or 0)
    nanos = float(stats.get("eval_duration") or 0.0)
    return count, (count / (nanos / 1e9) if nanos > 0 else 0.0)


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post random English math problems to the local model, one after another.",
        epilog="Press ESC or q to stop; Ctrl+C does the same.")
    p.add_argument("--model", help="model to ask (default: the loaded or only installed one)")
    p.add_argument("--count", type=int, default=0, metavar="N",
                   help="stop after N problems (default: keep going until ESC)")
    p.add_argument("--host", default=config.DEFAULT_HOST,
                   help=f"server address (default: {config.DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=config.DEFAULT_PORT,
                   help=f"server port (default: {config.DEFAULT_PORT})")
    p.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                   help="wait between problems (default: 0, ask immediately)")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="sampling temperature (default: 0.2 - maths wants a steady hand)")
    p.add_argument("--max-tokens", type=int, default=160,
                   help="answer length cap (default: 160 - a small model shows its working "
                        "however firmly it is asked not to, and 96 cut it off mid-sentence)")
    p.add_argument("--seed", type=int, help="seed for the problem generator (repeatable runs)")
    p.add_argument("--no-stream", action="store_true",
                   help="wait for the whole answer instead of printing it as it arrives")
    p.add_argument("--no-start", action="store_true",
                   help="fail instead of starting a server when none is running")
    p.add_argument("--keep-loaded", action="store_true",
                   help="leave the model in VRAM on exit (default: unload it)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    base = config.server_url(process.connect_host(args.host), args.port)
    client = httpx.Client(base_url=base, trust_env=False, timeout=600)
    spawned = None

    try:
        health = process.server_health(args.host, args.port)
        if health is None:
            if process.port_in_use(args.host, args.port):
                print(f"Port {args.port} on {args.host} is in use by something that is not "
                      f"an hfserve server.", file=sys.stderr)
                return 2
            if args.no_start:
                print(f"No server on {base}. Start one with:  hfserve serve", file=sys.stderr)
                return 2
            print(f"Starting the server on {base} ...")
            spawned = process.spawn_server(args.host, args.port)
            if not process.wait_for_server(args.host, args.port, proc=spawned):
                print("The server did not start.\n" + process.log_tail(10), file=sys.stderr)
                return 2
            health = process.server_health(args.host, args.port) or {}

        model = pick_model(client, args.model, health)
        options = {"temperature": args.temperature, "num_predict": args.max_tokens}

        stop = StopKey()
        stop.start()
        how_to_quit = "ESC or q to stop" if stop.watching else "Ctrl+C to stop"
        print(f"\nAsking {model} at {base} - {how_to_quit}.")
        print("The first question also loads the model, so it may take a while.\n")

        asked = correct = 0
        total_seconds = 0.0
        total_tokens = 0
        failures = 0

        while not stop.event.is_set() and (args.count <= 0 or asked < args.count):
            problem = make_problem(rng)
            asked += 1
            print(f"[{asked}] {problem.question}")

            payload = {"model": model, "options": options, "think": False,
                       "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": problem.question}]}
            started = time.perf_counter()
            try:
                if args.no_stream:
                    text, stats = ask_blocking(client, payload)
                    print(f"    -> {text.strip()}")
                else:
                    print("    -> ", end="", flush=True)
                    text, stats = ask_streaming(
                        client, payload, stop.event,
                        lambda d: print(d, end="", flush=True))
                    print()
            except (httpx.HTTPError, RuntimeError) as e:
                print(f"    !! {type(e).__name__}: {e}")
                failures += 1
                if failures >= 3:
                    print("\nThree failures in a row - stopping.", file=sys.stderr)
                    return 1
                continue
            except KeyboardInterrupt:
                print()
                stop.key = "Ctrl+C"
                stop.event.set()
                break
            failures = 0
            elapsed = time.perf_counter() - started

            if stop.event.is_set() and not text.strip():
                break  # stopped before the model said anything

            ok, given = check(text, problem.answer)
            correct += int(ok)
            total_seconds += elapsed
            tokens, rate = tokens_per_second(stats)
            total_tokens += tokens

            verdict = "correct" if ok else f"wrong (expected {_int(problem.answer)})"
            if given is None:
                verdict = f"no number in the reply (expected {_int(problem.answer)})"
            parts = [("OK  " if ok else "XX  ") + verdict, f"{elapsed:.1f} s"]
            if tokens:
                parts.append(f"{tokens} tokens")
                parts.append(f"{rate:.1f} tok/s")
            parts.append(problem.kind)
            # ASCII separators on purpose: a Windows console in cp437/cp850 cannot
            # print a middle dot and shows a replacement character instead.
            print("    " + " | ".join(parts))

            if args.delay > 0 and not stop.event.is_set():
                stop.event.wait(args.delay)  # interruptible, unlike sleep()

        reason = f"stopped by {stop.key}" if stop.key else "done"
        if asked:
            share = 100.0 * correct / asked
            average = total_seconds / asked
            print(f"\n{asked} problems | {correct} correct ({share:.0f}%) | "
                  f"{average:.1f} s each | {total_tokens} tokens | {reason}")
        else:
            print(f"\nNothing asked | {reason}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        if not args.keep_loaded:
            try:  # leave no weights in VRAM behind us; the next request reloads them
                client.post("/api/unload", timeout=120)
            except httpx.HTTPError:
                pass
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        if spawned is not None:
            # Started here, so stopped here; a server that was already up stays up.
            print("Stopping the server this script started ...")
            process.stop_server(args.host, args.port, proc=spawned, log=lambda *a, **k: None)


if __name__ == "__main__":
    raise SystemExit(main())
