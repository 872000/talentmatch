"""TalentMatch — resume ↔ job-description matching engine.

From-scratch TF-IDF + cosine similarity + phrase-based skill extraction.
Runs fully offline; the core engine depends only on the Python standard
library.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

_SKILLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.json")

with open(_SKILLS_PATH, encoding="utf-8") as _fh:
    SKILL_VOCAB: dict[str, list[str]] = json.load(_fh)

# Flat skill -> category lookup, all lower-cased.
SKILL_TO_CATEGORY: dict[str, str] = {}
for _cat, _skills in SKILL_VOCAB.items():
    for _s in _skills:
        SKILL_TO_CATEGORY[_s.lower()] = _cat


_DOT, _HASH, _PLUS = "\x00", "\x01", "\x02"


def _normalise(text: str) -> str:
    """Lower-case text; keep intra-word . # + (node.js, c#, c++).

    Sentence punctuation is stripped so a skill at the end of a sentence
    ("...on AWS.") still matches.
    """
    text = text.lower()
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", _DOT, text)  # node.js
    text = re.sub(r"(?<=[a-z0-9])#", _HASH, text)  # c#
    text = re.sub(r"(?<=[a-z0-9#+])\+", _PLUS, text)  # c++
    text = re.sub(r"[^a-z0-9\x00\x01\x02\s]", " ", text)
    return text.replace(_DOT, ".").replace(_HASH, "#").replace(_PLUS, "+")


def tokenise(text: str) -> list[str]:
    """Split normalised text into tokens."""
    return [t for t in _normalise(text).split() if t]


class TfIdf:
    """Minimal TF-IDF vectoriser fit on a document collection."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.vocab: list[str] = []

    def fit(self, documents: list[list[str]]) -> "TfIdf":
        n_docs = len(documents)
        df: dict[str, int] = {}
        for doc in documents:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.vocab = sorted(df)
        self.idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }
        return self

    def transform(self, tokens: list[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        total = max(len(tokens), 1)
        vec: dict[str, float] = {}
        for term, count in counts.items():
            if term in self.idf:
                vec[term] = (count / total) * self.idf[term]
        return vec


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0.0) for t in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_skills(text: str) -> set[str]:
    """Phrase-match the curated skill vocabulary against text.

    Matching happens on token sequences, so multi-word skills ("spring
    boot", "tailwind css") match naturally and a component word is never
    reported when only the longer phrase appears ("spring boot" does not
    imply bare "spring").
    """
    tokens = tokenise(text)
    n = len(tokens)
    occupied = [False] * n
    found: set[str] = set()
    # Longest phrases first so they claim their tokens before components.
    for skill in sorted(SKILL_TO_CATEGORY, key=lambda s: len(tokenise(s)), reverse=True):
        parts = tokenise(skill)
        k = len(parts)
        if k == 0 or k > n:
            continue
        for i in range(n - k + 1):
            if any(occupied[i : i + k]):
                continue
            if tokens[i : i + k] == parts:
                found.add(skill)
                for j in range(i, i + k):
                    occupied[j] = True
                break
    return found


@dataclass
class MatchResult:
    resume_name: str
    jd_name: str
    score: float  # 0-100
    cosine_similarity: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    coverage_by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    jd_skills: list[str] = field(default_factory=list)
    resume_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "resume": self.resume_name,
            "jd": self.jd_name,
            "score": round(self.score, 1),
            "cosine_similarity": round(self.cosine_similarity, 4),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "coverage_by_category": {
                cat: {k: round(v, 3) for k, v in vals.items()}
                for cat, vals in self.coverage_by_category.items()
            },
        }


def match(resume_text: str, jd_text: str, resume_name: str = "resume", jd_name: str = "jd") -> MatchResult:
    """Score a resume against a job description.

    Base signal: cosine similarity between TF-IDF vectors (fit on the two
    documents so the score is comparable per-pair). Skill matching refines
    the signal: the final 0-100 score blends textual similarity with the
    fraction of JD skills the resume covers.
    """
    resume_tokens = tokenise(resume_text)
    jd_tokens = tokenise(jd_text)

    vectoriser = TfIdf().fit([resume_tokens, jd_tokens])
    sim = cosine(vectoriser.transform(resume_tokens), vectoriser.transform(jd_tokens))

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)

    skill_coverage = len(matched) / max(len(jd_skills), 1)

    # Final score: blend text similarity (0..1) with skill coverage.
    score = round((0.55 * sim + 0.45 * skill_coverage) * 100, 1)

    coverage: dict[str, dict[str, float]] = {}
    for cat, skills in SKILL_VOCAB.items():
        cat_jd = [s for s in jd_skills if SKILL_TO_CATEGORY.get(s) == cat]
        if not cat_jd:
            continue
        cat_matched = [s for s in matched if SKILL_TO_CATEGORY.get(s) == cat]
        coverage[cat] = {
            "matched": len(cat_matched),
            "total": len(cat_jd),
            "coverage": len(cat_matched) / len(cat_jd),
        }

    return MatchResult(
        resume_name=resume_name,
        jd_name=jd_name,
        score=score,
        cosine_similarity=round(sim, 4),
        matched_skills=matched,
        missing_skills=missing,
        coverage_by_category=coverage,
        jd_skills=sorted(jd_skills),
        resume_skills=sorted(resume_skills),
    )


def rank(resume_texts: dict[str, str], jd_text: str, jd_name: str = "jd") -> list[MatchResult]:
    """Rank all resumes against one JD, best score first."""
    results = [
        match(text, jd_text, resume_name=name, jd_name=jd_name)
        for name, text in resume_texts.items()
    ]
    return sorted(results, key=lambda r: r.score, reverse=True)


def gap_analysis(result: MatchResult, target: float = 90.0) -> dict:
    """Report what skills to add so a resume would reach `target` score."""
    return {
        "current_score": result.score,
        "target": target,
        "gap": round(max(target - result.score, 0.0), 1),
        "add_these_skills": result.missing_skills,
        "skills_to_add_count": len(result.missing_skills),
        "verdict": (
            "Already at/above target." if result.score >= target
            else f"Add {len(result.missing_skills)} missing skill(s) to close the gap."
        ),
    }
