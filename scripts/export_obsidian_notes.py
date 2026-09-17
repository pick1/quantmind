#!/usr/bin/env python3
"""Generate Obsidian article notes from QuantMind DB and push to workhorse vault."""

import os
import sys
import tempfile
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantmind.store.sqlite import DocumentStore


CONTROLLED_VOCAB = [
    "factor-models", "risk-management", "nlp-finance",
    "portfolio-optimization", "market-microstructure",
    "alternative-data", "ml-methods", "macro", "sentiment",
]


def infer_tags(title: str, text: str) -> list[str]:
    """Basic keyword-based tag inference."""
    t = title.lower() + " " + (text or "").lower()
    tags = []
    rules = [
        ("time series", "ml-methods"),
        ("forecasting", "ml-methods"),
        ("prediction", "ml-methods"),
        ("deep learning", "ml-methods"),
        ("neural", "ml-methods"),
        ("transformer", "ml-methods"),
        ("temporal", "ml-methods"),
        ("multivariate", "ml-methods"),
        ("irregular", "ml-methods"),
        ("cluster", "ml-methods"),
        ("embedding", "ml-methods"),
        ("llm", "nlp-finance"),
        ("language model", "nlp-finance"),
        ("nlp", "nlp-finance"),
        ("financial", "nlp-finance"),
        ("financ", "nlp-finance"),
        ("sentiment", "sentiment"),
        ("macro", "macro"),
        ("risk", "risk-management"),
        ("volatility", "risk-management"),
        ("anomaly", "risk-management"),
        ("portfolio", "portfolio-optimization"),
        ("optimization", "portfolio-optimization"),
        ("factor", "factor-models"),
        ("factor model", "factor-models"),
        ("market", "market-microstructure"),
        ("microstructure", "market-microstructure"),
        ("alternative data", "alternative-data"),
        ("fund", "factor-models"),
        ("decision making", "nlp-finance"),
        ("multi-agent", "nlp-finance"),
        ("gqa", "ml-methods"),
        ("mla", "ml-methods"),
        ("car-following", "ml-methods"),
        ("traffic", "ml-methods"),
        ("power system", "macro"),
        ("fuzzy", "ml-methods"),
        ("poverty", "macro"),
        ("lake", "macro"),
        ("ecosystem", "macro"),
    ]
    for keyword, tag in rules:
        if keyword in t and len(tags) < 3:
            if tag not in tags:
                tags.append(tag)
    return tags if tags else ["ml-methods"]


def slugify(title: str) -> str:
    slug = title.lower().replace(" ", "-").replace(":", "")
    slug = "".join(c for c in slug if c.isalnum() or c in "-_")
    return slug[:60]


def build_note(doc: dict) -> str:
    doc_id = doc["id"]
    title = doc.get("title", "Untitled") or "Untitled"
    source_type = doc.get("source_type", "?")
    authors = doc.get("authors", "") or ""
    summary = doc.get("simple_summary", "") or ""
    abstract = doc.get("abstract", "") or ""

    tags = infer_tags(title, summary or abstract)

    source_map = {
        "arxiv": "arXiv",
        "pdf": "PDF Upload",
        "web": "Web Page",
        "text": "Text Input",
    }
    src_label = source_map.get(source_type, source_type)

    lines = ["---"]
    lines.append(f'title: {title}')
    lines.append("category: ingested")
    lines.append("confidence: 0.85")
    lines.append("date: '2026-06-15'")
    lines.append(f"source: {src_label} | quantmind-db-{doc_id}")
    lines.append("tags:")
    lines.append("  - article")
    lines.append(f"  - {source_type}")
    lines.append("  - ingested")
    for t in tags:
        lines.append(f"  - {t}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")

    if authors:
        lines.append(f"**Authors:** {authors}")
        lines.append("")
    lines.append(f"**Source:** {src_label} | QuantMind ID {doc_id}")
    lines.append("")

    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")

    if abstract and not summary:
        lines.append("## Abstract")
        lines.append("")
        lines.append(abstract[:500] + "...")
        lines.append("")

    lines.append("## Tags")
    lines.append("")
    for t in tags:
        lines.append(f"- {t}")
    lines.append("- article")
    lines.append("- ingested")
    lines.append("")
    lines.append("## Related")
    lines.append("")
    lines.append("- [[../quantmind-local|QuantMind Local]]")
    lines.append("- [[index|QuantMind Articles Index]]")
    lines.append("")

    return "\n".join(lines)


def build_index(docs: list) -> str:
    lines = ["---"]
    lines.append("title: QuantMind Articles Index")
    lines.append("tags:")
    lines.append("  - index")
    lines.append("  - active")
    lines.append("---")
    lines.append("")
    lines.append("# QuantMind Articles Index")
    lines.append("")
    lines.append(
        "Articles ingested via QuantMind knowledge extraction system. "
    )
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append("- [[../quantmind-local|QuantMind Local]]")
    lines.append("")
    lines.append("## Articles by Domain")
    lines.append("")

    # Group by tags
    domain_articles = {}
    for d in docs:
        title = d.get("title", "Untitled") or "Untitled"
        slug = slugify(title)
        t = infer_tags(title, d.get("simple_summary", "") or d.get("abstract", "") or "")
        for tag in t:
            domain_articles.setdefault(tag, []).append((slug, title))

    for tag in sorted(domain_articles.keys()):
        lines.append(f"### {tag}")
        for slug, title in domain_articles[tag]:
            lines.append(f"- [[{slug}|{title}]]")
        lines.append("")

    lines.append("## All Articles")
    lines.append("")
    for d in docs:
        title = d.get("title", "Untitled") or "Untitled"
        slug = slugify(title)
        lines.append(f"- [[{slug}|{title}]]")
    lines.append("")

    return "\n".join(lines)


def main():
    store = DocumentStore()
    docs = store.list_documents(limit=100)
    print(f"Found {len(docs)} documents")

    notes_dir = tempfile.mkdtemp(prefix="qm_obsidian_")

    for d in docs:
        content = build_note(d)
        slug = slugify(d.get("title", "untitled") or "untitled")
        fpath = os.path.join(notes_dir, f"{slug}.md")
        with open(fpath, "w") as f:
            f.write(content)

    # Write index
    idx_content = build_index(docs)
    with open(os.path.join(notes_dir, "index.md"), "w") as f:
        f.write(idx_content)

    print(f"Notes written to {notes_dir}")

    # Push to workhorse via tar-pipe (handles spaces in paths)
    pw = "Eromitlab1!"
    subprocess.run(
        [
            "sshpass", "-p", pw,
            "ssh", "-o", "StrictHostKeyChecking=accept-new",
            "workhorse@192.168.1.193",
            "mkdir -p '/home/workhorse/Documents/Obsidian Vault/knowledge/quantmind/'",
        ],
        check=True,
    )
    # Tar-pipe the notes over to avoid path-space issues with scp
    tar_proc = subprocess.Popen(
        ["tar", "-C", notes_dir, "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        [
            "sshpass", "-p", pw,
            "ssh", "-o", "StrictHostKeyChecking=accept-new",
            "workhorse@192.168.1.193",
            "tar -C '/home/workhorse/Documents/Obsidian Vault/knowledge/quantmind/' -xf -",
        ],
        stdin=tar_proc.stdout,
        check=True,
    )
    tar_proc.wait()

    shutil.rmtree(notes_dir)
    print("Done — notes pushed to workhorse vault")


if __name__ == "__main__":
    main()
