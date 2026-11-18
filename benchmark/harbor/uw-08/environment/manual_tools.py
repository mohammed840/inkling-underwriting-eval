#!/usr/bin/env python3
"""Deterministic lexical retrieval over the pinned underwriting source corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    words = ["".join(c for c in word if not unicodedata.combining(c)) for word in TOKEN_RE.findall(normalized)]
    terms = []
    for word in words:
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        else:
            for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
                if len(word) > len(suffix) + 3 and word.endswith(suffix):
                    word = word[: -len(suffix)]
                    if suffix == "ing" and len(word) > 2 and word[-1] == word[-2]:
                        word = word[:-1]
                    break
        terms.append(word)
    return terms


class ManualTools:
    """Search and read source sections from a corpus.json file."""

    def __init__(self, corpus_path: str | Path):
        self.corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        sources = self.corpus.get("sources")
        sections = self.corpus.get("sections")
        if not isinstance(sources, list) or not isinstance(sections, list):
            raise ValueError("corpus must contain sources and sections arrays")
        self.source_ids = {row.get("source_id") for row in sources if isinstance(row, dict)}
        self.sections: dict[str, dict[str, dict]] = defaultdict(dict)
        self.token_counts: dict[str, Counter[str]] = {}
        self.document_frequency: Counter[str] = Counter()
        for row in sections:
            if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("source_id", "section_id", "text")):
                raise ValueError("each corpus section needs source_id, section_id, and text strings")
            source_id, section_id = row["source_id"], row["section_id"]
            if source_id not in self.source_ids or not section_id.startswith(source_id + ":"):
                raise ValueError(f"invalid section identity: {section_id}")
            if section_id in self.sections[source_id]:
                raise ValueError(f"duplicate section id: {section_id}")
            self.sections[source_id][section_id] = row
            terms = Counter(_terms(row["text"]))
            self.token_counts[section_id] = terms
            self.document_frequency.update(terms.keys())
        self.section_count = len(sections)

    def _require_source(self, source_id: str) -> None:
        if source_id not in self.source_ids:
            raise ValueError(f"unknown source_id: {source_id}")

    def search_manual(self, query: str, source_id: str, limit: int = 5) -> list[dict]:
        self._require_source(source_id)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        query_terms = set(_terms(query))
        if not query_terms:
            return []
        candidates = []
        for section_id, row in self.sections[source_id].items():
            counts = self.token_counts[section_id]
            matched = query_terms.intersection(counts)
            if not matched:
                continue
            score = sum(
                math.log1p(self.section_count / self.document_frequency[term])
                * (1 + math.log(counts[term]))
                for term in matched
            )
            candidates.append((score, section_id, row))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [dict(row, score=round(score, 6)) for score, _, row in candidates[:limit]]

    def read_section(self, source_id: str, section_id: str) -> dict:
        self._require_source(source_id)
        try:
            return dict(self.sections[source_id][section_id])
        except KeyError as exc:
            raise ValueError(f"unknown section_id for {source_id}: {section_id}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search_manual", help="rank matching source sections")
    search.add_argument("--source-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5)
    read = commands.add_parser("read_section", help="return one source section")
    read.add_argument("--source-id", required=True)
    read.add_argument("--section-id", required=True)
    args = parser.parse_args()
    try:
        tools = ManualTools(Path(__file__).with_name("corpus.json"))
        result = (
            tools.search_manual(args.query, args.source_id, args.limit)
            if args.command == "search_manual"
            else tools.read_section(args.source_id, args.section_id)
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
