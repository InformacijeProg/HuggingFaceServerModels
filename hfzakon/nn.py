"""narodne-novine.nn.hr: mirror the official gazette (sluzbeni dio) per year and chunk it.

Discovery walks the search pages (godina/broj, rpp=200) exactly like the site's own
"Po broju" form; each document page carries its own metadata block (vrsta, donositelj,
ELI, datum) which ends up in every chunk's header and metadata.
"""
from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from . import config, fetch
from .zakon_hr import CLANAK_RE, _split_paragraphs

DOC_LINK_RE = re.compile(r'href="(/clanci/sluzbeni/[^"]+\.html)"')
EDITION_RE = re.compile(r"/eli/sluzbeni/(\d{4})/(\d+)/pdf")
PRINT_TITLE_RE = re.compile(r"NN\s+(\d+)/(\d{4})\s*\(([^)]*)\)")


def editions(year: int) -> int:
    """Highest sluzbeni-dio edition number for the year, read live from the search page."""
    data = fetch.get(config.NN_SEARCH.format(year=year, broj=1))
    html = data.decode("utf-8", "replace")
    marker = f'class="all-edition-{year}-1"'
    i = html.find(marker)
    if i < 0:
        return 0
    j = html.find("all-edition-", i + len(marker))
    block = html[i:j if j > 0 else len(html)]
    nums = [int(m.group(2)) for m in EDITION_RE.finditer(block) if int(m.group(1)) == year]
    return max(nums) if nums else 0


def search_cache(year: int, broj: int):
    return config.RAW_NN_SEARCH_DIR / f"{year}_{broj:03d}.html"


def doc_cache(url_path: str):
    return config.RAW_NN_DOC_DIR / url_path.rsplit("/", 1)[-1]


def doc_links(search_html: bytes) -> list[str]:
    seen: list[str] = []
    for m in DOC_LINK_RE.finditer(search_html.decode("utf-8", "replace")):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def pull(years=config.NN_YEARS, progress=print) -> tuple[int, int]:
    """Download all search pages and document pages not yet cached. Returns (fetched, total docs)."""
    config.ensure_dirs()
    fetched = total = 0
    for year in years:
        last = editions(year)
        if progress:
            progress(f"  NN {year}: editions 1..{last}")
        for broj in range(1, last + 1):
            page = fetch.get_cached(config.NN_SEARCH.format(year=year, broj=broj),
                                    search_cache(year, broj))
            if not page:
                continue
            for path in doc_links(page):
                url = config.NN_BASE + path
                if fetch.nn_disallowed(url):
                    continue
                total += 1
                cached = doc_cache(path).exists()
                if fetch.get_cached(url, doc_cache(path)) and not cached:
                    fetched += 1
            if progress and broj % 10 == 0:
                progress(f"  NN {year} broj {broj}/{last} ({total} docs, {fetched} new)")
    if progress:
        progress(f"  NN done: {total} documents ({fetched} new)")
    return fetched, total


def parse_doc(html: bytes, url: str) -> dict | None:
    """Title, gazette metadata and body paragraphs of one document page."""
    soup = BeautifulSoup(html, "lxml")
    frame = soup.find(id="html-content-frame")
    if frame is None:
        return None
    sticky = soup.find(id="sticky")
    h2 = sticky.find("h2") if sticky else None
    title = h2.get_text(" ", strip=True) if h2 else (soup.title.get_text(strip=True) if soup.title else "")

    issue = date = ""
    print_title = frame.find("span", class_="title-for-print")
    if print_title:
        m = PRINT_TITLE_RE.search(print_title.get_text(" ", strip=True))
        if m:
            issue, date = f"{m.group(1)}/{m.group(2)}", m.group(3)
        print_title.extract()
    for el in frame.find_all(["style", "script"]):
        el.extract()

    meta = {"vrsta": "", "donositelj": "", "eli": ""}
    side = soup.find("div", class_="metadata")
    if side:
        for row in side.find_all(["tr", "div"]):
            t = row.get_text(" ", strip=True)
            if t.startswith("Vrsta dokumenta:") and not meta["vrsta"]:
                meta["vrsta"] = t.split(":", 1)[1].strip()
            elif t.startswith("Donositelj:") and not meta["donositelj"]:
                meta["donositelj"] = t.split(":", 1)[1].strip()
        eli = side.find("a", href=re.compile(r"/eli/sluzbeni/\d{4}/\d+/\d+$"))
        if eli:
            meta["eli"] = eli.get("href", "")

    paragraphs: list[str] = []
    for el in frame.find_all(["p", "h2", "h3", "h4", "table", "li"]):
        if el.find_parent("table") is not None and el.name != "table":
            continue
        if el.name == "table":
            for tr in el.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                line = " | ".join(c for c in cells if c)
                if line:
                    paragraphs.append(line)
            continue
        text = el.get_text(" ", strip=True).replace("\xa0", " ").strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        return None
    return {"url": url, "title": title, "issue": issue, "date": date,
            "vrsta": meta["vrsta"], "donositelj": meta["donositelj"], "eli": meta["eli"],
            "paragraphs": paragraphs}


def chunk_doc(doc: dict, doc_key: str):
    """Yield (id, text, meta). Law-like docs split per Clanak, prose splits into windows."""
    header = doc["title"]
    sub = ", ".join(x for x in (f"NN {doc['issue']}" if doc["issue"] else "",
                                doc["date"], doc["vrsta"], doc["donositelj"]) if x)
    if sub:
        header += f"\n{sub}"
    meta = {"source": "nn", "title": doc["title"][:300], "issue": doc["issue"],
            "date": doc["date"], "vrsta": doc["vrsta"], "donositelj": doc["donositelj"],
            "eli": doc["eli"], "url": doc["url"]}

    paras = doc["paragraphs"]
    markers = [i for i, p in enumerate(paras) if CLANAK_RE.match(p)]
    if len(markers) >= 2:
        pieces: list[tuple[str, list[str]]] = []
        if markers[0] > 0:
            pieces.append(("uvod", paras[:markers[0]]))
        for k, start in enumerate(markers):
            end = markers[k + 1] if k + 1 < len(markers) else len(paras)
            m = CLANAK_RE.match(paras[start])
            num = m.group(1) + (m.group(2) or "")
            pieces.append((f"cl{num}", paras[start:end]))
        seen: dict[str, int] = {}
        for key, part_paras in pieces:
            n = seen[key] = seen.get(key, 0) + 1
            kid = f"nn:{doc_key}:{key}" + (f"-{n}" if n > 1 else "")
            body = "\n".join(part_paras)
            if len(header) + len(body) <= config.MAX_CHUNK_CHARS:
                yield kid, f"{header}\n{body}", meta
            else:
                for k2, part in enumerate(_split_paragraphs(part_paras, config.TARGET_PART_CHARS), 1):
                    yield f"{kid}:p{k2}", f"{header}\n" + "\n".join(part), dict(meta, part=str(k2))
        return

    parts = _split_paragraphs(paras, config.TARGET_PART_CHARS)
    for k, part in enumerate(parts, 1):
        suffix = dict(meta, part=str(k)) if len(parts) > 1 else meta
        yield f"nn:{doc_key}:c{k}", f"{header}\n" + "\n".join(part), suffix


def chunk_all(years=config.NN_YEARS, progress=print) -> int:
    """Parse every cached document and rewrite chunks/nn_<year>.chunks.jsonl per year."""
    config.ensure_dirs()
    total = 0
    for year in years:
        out_path = config.CHUNKS_DIR / config.NN_CHUNKS_TPL.format(year=year)
        n_docs = n_chunks = 0
        with out_path.open("w", encoding="utf-8") as out:
            for path in sorted(config.RAW_NN_DOC_DIR.glob(f"{year}_*.html")):
                html = path.read_bytes()
                if not html:
                    continue
                doc = parse_doc(html, f"{config.NN_BASE}/clanci/sluzbeni/{path.name}")
                if not doc:
                    continue
                for cid, text, meta in chunk_doc(doc, path.stem):
                    if len(text) < config.MIN_CHUNK_CHARS:
                        continue
                    out.write(json.dumps({"id": cid, "text": text, "meta": meta},
                                         ensure_ascii=False) + "\n")
                    n_chunks += 1
                n_docs += 1
                if progress and n_docs % 500 == 0:
                    progress(f"  NN {year}: {n_docs} docs -> {n_chunks:,} chunks")
        total += n_chunks
        if progress:
            progress(f"  NN {year}: {n_docs} docs -> {n_chunks:,} chunks -> {out_path}")
    return total
