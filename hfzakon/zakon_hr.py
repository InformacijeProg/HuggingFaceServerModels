"""zakon.hr: every consolidated law from the sitemap, parsed into per-article chunks.

Each /z/<id>/<slug> page is the latest consolidated text of one law, so "the same law
in multiple years" resolves itself: there is exactly one current page per law.
"""
from __future__ import annotations

import json
import re
import urllib.parse

from bs4 import BeautifulSoup

from . import config, fetch

# "Članak 5." / "Članak 5.a" and the pre-1991 form "Član 5." used by old laws.
CLANAK_RE = re.compile(r"^\s*Član(?:ak)?\s+(\d+)\.?\s*([a-zčćđšž])?\.?\s*$", re.IGNORECASE)
CLANAK_INLINE_RE = re.compile(r"(?=\bČlan(?:ak)?\s+\d+[a-zčćđšž]?\.(?:\s|$))")
STAVAK_RE = re.compile(r"^\((\d+)\)")


def law_urls(refresh: bool = False) -> list[str]:
    """All /z/ law URLs from the sitemap, cached in state/zakon_urls.txt."""
    config.ensure_dirs()
    cache = config.STATE_DIR / "zakon_urls.txt"
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8").splitlines()
    data = fetch.get(config.ZAKON_SITEMAP)
    subs = re.findall(r"<loc>(.*?)</loc>", data.decode("utf-8", "replace"))
    urls: set[str] = set()
    for sub in subs:
        page = fetch.get(sub)
        if not page:
            continue
        for loc in re.findall(r"<loc>(.*?)</loc>", page.decode("utf-8", "replace")):
            if re.search(r"/z/\d+/", loc):
                urls.add(loc)
    out = sorted(urls, key=lambda u: int(u.split("/z/")[1].split("/")[0]))
    cache.write_text("\n".join(out), encoding="utf-8")
    return out


def law_id(url: str) -> str:
    return url.split("/z/")[1].split("/")[0]


ARCHIVED_SLUG_RE = re.compile(r"-\d{4}-\d{4}/?$")


def discover_ids(progress=print, stop_after_misses: int = 60) -> int:
    """Probe /z/<id>/x for ids the stale sitemap misses (holes + beyond the max).

    Real law pages are cached like pulled ones; homepage redirects mean the id does
    not exist; /c/ pages with a -YYYY-YYYY slug are superseded versions of laws whose
    current text is already indexed, so they are skipped. Resumable via state file.
    Returns how many new law pages were found.
    """
    import time

    config.ensure_dirs()
    known = {int(law_id(u)) for u in law_urls()}
    checked_path = config.STATE_DIR / "zakon_ids_checked.txt"
    checked: dict[int, str] = {}
    if checked_path.exists():
        for line in checked_path.read_text(encoding="utf-8").splitlines():
            i, _, result = line.partition("\t")
            checked[int(i)] = result
    extra_path = config.STATE_DIR / "zakon_urls_extra.txt"
    extra = extra_path.read_text(encoding="utf-8").splitlines() if extra_path.exists() else []
    claimed = {l.split("\t")[1] for l in extra if "\t" in l}

    lo, hi = min(known), max(known)
    candidates = [i for i in range(lo, hi + 1) if i not in known] + list(range(hi + 1, hi + 2000))
    found = probed = 0
    misses = 0
    with checked_path.open("a", encoding="utf-8") as log, extra_path.open("a", encoding="utf-8") as out:
        for i in candidates:
            if i > hi and misses >= stop_after_misses:
                break
            if i in checked or i in known:
                continue
            try:
                time.sleep(config.DELAY_S)
                r = fetch.session().get(f"{config.ZAKON_BASE}/z/{i}/x", timeout=config.TIMEOUT_S)
            except Exception:  # noqa: BLE001 - transient; re-probed on the next run
                continue
            probed += 1
            final = r.url
            if r.status_code != 200 or final.rstrip("/") == config.ZAKON_BASE or "/z/" not in final and "/c/" not in final:
                result = "miss"
                if i > hi:
                    misses += 1
            elif "/c/" in final and ARCHIVED_SLUG_RE.search(final):
                result = "archived"
            elif final in claimed:
                result = "dup"
            else:
                (config.RAW_ZAKON_DIR / "laws" / f"{i}.html").write_bytes(r.content)
                out.write(f"{i}\t{final}\n")
                out.flush()
                claimed.add(final)
                result = "law"
                found += 1
                if i > hi:
                    misses = 0
            log.write(f"{i}\t{result}\n")
            log.flush()
            if progress and probed % 100 == 0:
                progress(f"  probed {probed} ids ({found} new laws)")
    if progress:
        progress(f"  id discovery: {probed} probed, {found} new law pages")
    return found


def law_cache_path(url: str):
    return config.RAW_ZAKON_DIR / "laws" / f"{law_id(url)}.html"


def discover_new(progress=None, stop_after_misses: int = 40) -> list[int]:
    """Tail-only id probe for the refresh path: new laws always get higher ids.

    Starts above the highest id ever seen as a real page (law/archived/dup), so ids
    that were misses last time are re-checked — a miss may become a law later.
    Returns the ids of newly cached law pages.
    """
    import time

    config.ensure_dirs()
    known = {int(law_id(u)) for u in law_urls()}
    top = max(known)
    checked_path = config.STATE_DIR / "zakon_ids_checked.txt"
    if checked_path.exists():
        for line in checked_path.read_text(encoding="utf-8").splitlines():
            i, _, result = line.partition("\t")
            if result != "miss" and i.isdigit():
                top = max(top, int(i))
    extra_path = config.STATE_DIR / "zakon_urls_extra.txt"
    claimed = set()
    if extra_path.exists():
        claimed = {l.split("\t")[1] for l in extra_path.read_text(encoding="utf-8").splitlines()
                   if "\t" in l}

    found: list[int] = []
    misses = 0
    i = top
    with checked_path.open("a", encoding="utf-8") as log, extra_path.open("a", encoding="utf-8") as out:
        while misses < stop_after_misses:
            i += 1
            try:
                time.sleep(config.DELAY_S)
                r = fetch.session().get(f"{config.ZAKON_BASE}/z/{i}/x", timeout=config.TIMEOUT_S)
            except Exception:  # noqa: BLE001 - transient; next refresh retries
                misses += 1
                continue
            final = r.url
            if (r.status_code != 200 or final.rstrip("/") == config.ZAKON_BASE
                    or ("/z/" not in final and "/c/" not in final)):
                result = "miss"
                misses += 1
            elif "/c/" in final and ARCHIVED_SLUG_RE.search(final):
                result, misses = "archived", 0
            elif final in claimed:
                result, misses = "dup", 0
            else:
                (config.RAW_ZAKON_DIR / "laws" / f"{i}.html").write_bytes(r.content)
                out.write(f"{i}\t{final}\n")
                out.flush()
                claimed.add(final)
                result, misses = "law", 0
                found.append(i)
                if progress:
                    progress(f"  new law id {i}: {final}")
            log.write(f"{i}\t{result}\n")
            log.flush()
    return found


def pull(limit: int | None = None, progress=print, refetch_empty: bool = False) -> tuple[int, int]:
    """Download every law page not yet cached. Returns (fetched, total).

    refetch_empty drops cached pages that parse to no articles first: zakon.hr
    occasionally serves degraded pages under load, a slower re-fetch recovers them.
    """
    urls = law_urls()
    if limit:
        urls = urls[:limit]
    if refetch_empty:
        dropped = 0
        for url in urls:
            path = law_cache_path(url)
            if not path.exists():
                continue
            html = path.read_bytes()
            law = parse_law(html, url) if html else None
            if not law or not law["articles"]:
                path.unlink()
                dropped += 1
        if progress:
            progress(f"  re-fetching {dropped} empty/unparsed pages")
    fetched = 0
    for i, url in enumerate(urls, 1):
        cached = law_cache_path(url).exists()
        if fetch.get_cached(url, law_cache_path(url)) and not cached:
            fetched += 1
        if progress and i % 50 == 0:
            progress(f"  zakon.hr {i}/{len(urls)} pages ({fetched} new)")
    if progress:
        progress(f"  zakon.hr done: {len(urls)} pages ({fetched} new)")
    return fetched, len(urls)


def parse_law(html: bytes, url: str) -> dict | None:
    """Title, NN references, status and the ordered list of articles of one law page.

    Handles both layouts: /z/ consolidated pages (cms-zakon-* classes) and the /c/zakon
    pages some /z/ URLs redirect to (plain p/h3 markup, old laws and expired versions).
    """
    soup = BeautifulSoup(html, "lxml")
    header = soup.find("div", class_="main-zakon-header")
    body = soup.find(id="pretrazitekst") or soup.find("div", class_="cms-text-content")
    if body is None:
        return None
    h1 = (header or soup).find("h1")
    title = h1.get_text(" ", strip=True) if h1 else urllib.parse.unquote(url.rsplit("/", 1)[-1])
    nn_refs, effective = "", ""
    head_ps = header.find_all("p") if header else reversed(body.find_all_previous("p"))
    for p in head_ps:
        t = p.get_text(" ", strip=True)
        if (t.startswith("NN") or re.match(r"^\d+/\d{2}\b", t)) and not nn_refs:
            nn_refs = t if t.startswith("NN") else f"NN {t}"
        if "na snazi" in t.lower() and not effective:
            m = re.search(r"na snazi.*", t, re.IGNORECASE)
            effective = m.group(0)
            nn_part = t[:m.start()].strip(" ,")
            if not nn_refs and re.match(r"^\d+/\d{2}\b", nn_part):
                nn_refs = f"NN {nn_part}"
    status_el = soup.find("span", class_="status-zakona")
    status = status_el.get_text(" ", strip=True) if status_el else ""

    articles: list[dict] = []
    preamble: list[str] = []
    section = ""
    current: dict | None = None
    for el in body.find_all(["h3", "p"]):
        text = el.get_text(" ", strip=True).replace("\xa0", " ").strip()
        if not text:
            continue
        classes = el.get("class") or []
        if el.name == "h3":
            section = text
            continue
        m = CLANAK_RE.match(text)
        if m:  # marker paragraphs carry cms-zakon-clanak on /z/ pages, no class on /c/ pages
            num = m.group(1) + (m.group(2) or "")
            current = {"number": num, "title": "", "section": section, "paragraphs": []}
            articles.append(current)
            continue
        if "cms-zakon-clanak" in classes:
            if current is not None and not current["paragraphs"] and not current["title"]:
                current["title"] = text
            elif current is None:
                preamble.append(text)
            continue
        if current is None:
            preamble.append(text)
        else:
            current["paragraphs"].append(text)
    if not articles:  # oldest laws: one text blob with inline "Član N." markers
        blob = "\n".join(preamble)
        pieces = CLANAK_INLINE_RE.split(blob)
        if len(pieces) >= 3:
            preamble = [pieces[0].strip()] if pieces[0].strip() else []
            articles = []
            for piece in pieces[1:]:
                mm = re.match(r"\s*Član(?:ak)?\s+(\d+)\.?\s*([a-zčćđšž])?\.?", piece)
                rest = piece[mm.end():].strip()
                articles.append({"number": mm.group(1) + (mm.group(2) or ""), "title": "",
                                 "section": "", "paragraphs": [rest] if rest else []})
    try:
        lid = law_id(url)
    except IndexError:  # discovered laws may live on /c/ pages; the caller sets the id
        lid = ""
    return {"id": lid, "url": url, "title": title, "nn_refs": nn_refs,
            "effective": effective, "status": status, "preamble": preamble,
            "articles": articles}


def _split_paragraphs(paragraphs: list[str], budget: int) -> list[list[str]]:
    """Greedy split on paragraph (stavak) boundaries; oversized single paragraphs split on sentences."""
    flat: list[str] = []
    for p in paragraphs:
        if len(p) <= config.MAX_CHUNK_CHARS:
            flat.append(p)
            continue
        piece = ""
        for sent in re.split(r"(?<=[.;:])\s+", p):
            if piece and len(piece) + len(sent) + 1 > budget:
                flat.append(piece)
                piece = sent
            else:
                piece = f"{piece} {sent}".strip()
        if piece:
            flat.append(piece)
    parts: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for p in flat:
        if cur and size + len(p) + 1 > budget:
            parts.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p) + 1
    if cur:
        parts.append(cur)
    return parts


def _part_label(paras: list[str]) -> str:
    nums = [m.group(1) for p in paras if (m := STAVAK_RE.match(p))]
    if not nums:
        return ""
    if nums[0] == nums[-1] or int(nums[-1]) < int(nums[0]):  # nested numbering can restart
        return f" st. {nums[0]}."
    return f" st. {nums[0]}.–{nums[-1]}."


def chunk_law(law: dict):
    """Yield (id, text, meta): one header chunk per law, one chunk per article (split by stavak)."""
    base_meta = {"source": "zakon.hr", "law": law["title"], "law_id": law["id"],
                 "url": law["url"], "nn_refs": law["nn_refs"], "status": law["status"],
                 "effective": law["effective"]}
    sections = []
    for a in law["articles"]:
        if a["section"] and a["section"] not in sections:
            sections.append(a["section"])
    head_lines = [law["title"]]
    sub = ", ".join(x for x in (law["nn_refs"], law["status"], law["effective"]) if x)
    if sub:
        head_lines.append(sub)
    if law["preamble"]:
        head_lines.append(" ".join(law["preamble"])[:400])
    if sections:
        head_lines.append("Sadržaj: " + "; ".join(sections)[:600])
    yield f"zakon:{law['id']}:head", "\n".join(head_lines), dict(base_meta, clanak="", clanak_title="")

    # Expired or repealed texts say so in every chunk, so answers cannot pass old law off as current.
    suffix = ""
    m = re.search(r"\bdo\s+(\d[\d.]*\.?)", law["effective"])
    if m:
        suffix = f" (na snazi do {m.group(1)})"
    elif law["status"] and law["status"].lower() != "na snazi":
        suffix = f" ({law['status']})"

    seen: dict[str, int] = {}
    for a in law["articles"]:
        n = seen[a["number"]] = seen.get(a["number"], 0) + 1
        cid = f"zakon:{law['id']}:cl{a['number']}" + (f"-{n}" if n > 1 else "")
        header = f"{law['title']}{suffix} — Članak {a['number']}."
        if a["title"]:
            header += f" ({a['title']})"
        meta = dict(base_meta, clanak=a["number"], clanak_title=a["title"], section=a["section"])
        body = "\n".join(a["paragraphs"])
        if not body:
            continue
        if len(header) + len(body) <= config.MAX_CHUNK_CHARS:
            yield cid, f"{header}\n{body}", meta
            continue
        parts = _split_paragraphs(a["paragraphs"], config.TARGET_PART_CHARS)
        for k, paras in enumerate(parts, 1):
            label = _part_label(paras) or f" (dio {k}/{len(parts)})"
            yield f"{cid}:p{k}", f"{header}{label}\n" + "\n".join(paras), dict(meta, part=str(k))


def chunk_all(progress=print) -> int:
    """Parse every cached law page and rewrite chunks/zakon_hr.chunks.jsonl."""
    config.ensure_dirs()
    n_laws = n_chunks = 0
    skipped: list[str] = []
    urls = {law_id(u): u for u in law_urls()}
    extra_path = config.STATE_DIR / "zakon_urls_extra.txt"
    if extra_path.exists():
        for line in extra_path.read_text(encoding="utf-8").splitlines():
            i, _, final = line.partition("\t")
            urls.setdefault(i, final)
    with config.ZAKON_CHUNKS.open("w", encoding="utf-8") as out:
        for path in sorted((config.RAW_ZAKON_DIR / "laws").glob("*.html")):
            html = path.read_bytes()
            if not html:
                continue
            url = urls.get(path.stem, f"{config.ZAKON_BASE}/z/{path.stem}/x")
            law = parse_law(html, url)
            if not law or not law["articles"]:
                skipped.append(path.stem)
                continue
            law["id"] = path.stem
            for cid, text, meta in chunk_law(law):
                if len(text) < config.MIN_CHUNK_CHARS:
                    continue
                out.write(json.dumps({"id": cid, "text": text, "meta": meta},
                                     ensure_ascii=False) + "\n")
                n_chunks += 1
            n_laws += 1
            if progress and n_laws % 100 == 0:
                progress(f"  {n_laws} laws -> {n_chunks:,} chunks")
    if skipped and progress:
        progress(f"  skipped (no articles found): {len(skipped)}: {', '.join(skipped[:20])}")
    if progress:
        progress(f"  zakon.hr chunks: {n_laws} laws -> {n_chunks:,} chunks -> {config.ZAKON_CHUNKS}")
    return n_chunks
