#!/usr/bin/env python3
"""TalentMatch CLI.

Usage:
    python -m talentmatch.cli match <resume> <jd>
    python -m talentmatch.cli rank <jd>
    python -m talentmatch.cli gaps <resume> <jd> [--target 90]
    python -m talentmatch.cli demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import gap_analysis, match, rank

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _resume_path(name: str) -> str:
    return os.path.join(DATA_DIR, "resumes", name if name.endswith(".md") else name + ".md")


def _jd_path(name: str) -> str:
    return os.path.join(DATA_DIR, "jds", name if name.endswith(".md") else name + ".md")


def _all_resumes() -> dict[str, str]:
    resumes = {}
    folder = os.path.join(DATA_DIR, "resumes")
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".md"):
            resumes[fname[:-3]] = _read(os.path.join(folder, fname))
    return resumes


def _bar(score: float, width: int = 30) -> str:
    filled = int(round(score / 100 * width))
    return "█" * filled + "░" * (width - filled)


def print_report(result, target: float = 90.0) -> None:
    r = result
    print(f"\n{'=' * 64}")
    print(f"  TALENTMATCH REPORT — {r.resume_name}  vs  {r.jd_name}")
    print(f"{'=' * 64}")
    print(f"\n  Match score : {r.score:5.1f}/100  {_bar(r.score)}")
    print(f"  Cosine sim  : {r.cosine_similarity:.4f}")
    print(f"\n  Matched skills ({len(r.matched_skills)}):")
    print(f"    {', '.join(r.matched_skills) if r.matched_skills else '—'}")
    print(f"\n  Missing skills ({len(r.missing_skills)}):")
    print(f"    {', '.join(r.missing_skills) if r.missing_skills else '— (none!)'}")
    print("\n  Coverage by category:")
    for cat, vals in r.coverage_by_category.items():
        print(f"    {cat:14} {vals['matched']:>2}/{vals['total']:<2}  ({vals['coverage'] * 100:5.1f}%)")
    gaps = gap_analysis(r, target=target)
    print(f"\n  Gap analysis → target {target:.0f}:")
    print(f"    {gaps['verdict']}")
    if gaps["add_these_skills"]:
        print(f"    Add: {', '.join(gaps['add_these_skills'])}")
    print()


def cmd_match(args: argparse.Namespace) -> int:
    result = match(
        _read(_resume_path(args.resume)),
        _read(_jd_path(args.jd)),
        resume_name=args.resume,
        jd_name=args.jd,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_report(result, target=args.target)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    jd_text = _read(_jd_path(args.jd))
    results = rank(_all_resumes(), jd_text, jd_name=args.jd)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    print(f"\n  Ranking for JD: {args.jd}")
    print(f"  {'#':<3} {'Candidate':<22} {'Score':>7}   Match")
    print("  " + "-" * 58)
    for i, r in enumerate(results, 1):
        print(f"  {i:<3} {r.resume_name:<22} {r.score:>6.1f}   {_bar(r.score, 24)}")
    print()
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    result = match(
        _read(_resume_path(args.resume)),
        _read(_jd_path(args.jd)),
        resume_name=args.resume,
        jd_name=args.jd,
    )
    gaps = gap_analysis(result, target=args.target)
    print(f"\n  Gap analysis: {args.resume} → {args.jd}")
    print(f"  Current score : {gaps['current_score']:.1f}/100")
    print(f"  Target        : {gaps['target']:.0f}")
    print(f"  Gap           : {gaps['gap']:.1f} points")
    print(f"  Verdict       : {gaps['verdict']}")
    if gaps["add_these_skills"]:
        print(f"\n  To reach {args.target:.0f}%, add these skills:")
        for s in gaps["add_these_skills"]:
            print(f"    • {s}")
    print()
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    print("\n  TALENTMATCH — end-to-end demo")
    print("  " + "-" * 58)
    # 1. Best pairing: backend java resume vs backend java JD.
    resumes = _all_resumes()
    jd = _read(_jd_path("backend_java_jd"))
    results = rank(resumes, jd, jd_name="backend_java_jd")
    print_report(results[0])
    # 2. Rank table for the frontend JD.
    print("\n  Ranking every candidate for frontend_react_jd:")
    fe_results = rank(resumes, _read(_jd_path("frontend_react_jd")), jd_name="frontend_react_jd")
    print(f"  {'Candidate':<22} {'Score':>7}")
    for r in fe_results:
        print(f"  {r.resume_name:<22} {r.score:>6.1f}")
    # 3. Gap analysis on a near-miss.
    print("\n  Gap analysis (frontend_vue → frontend_react_jd):")
    vue = match(resumes["frontend_vue"], _read(_jd_path("frontend_react_jd")),
                resume_name="frontend_vue", jd_name="frontend_react_jd")
    gaps = gap_analysis(vue, target=90.0)
    print(f"  Current: {gaps['current_score']:.1f} → target {gaps['target']:.0f}")
    print(f"  {gaps['verdict']}")
    print(f"  Add: {', '.join(gaps['add_these_skills'])}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="talentmatch", description="Resume ↔ JD matching engine.")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("match", help="Score one resume against one JD.")
    m.add_argument("resume", help="Resume name, e.g. backend_java")
    m.add_argument("jd", help="JD name, e.g. backend_java_jd")
    m.add_argument("--target", type=float, default=90.0)
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_match)

    r = sub.add_parser("rank", help="Rank all bundled resumes for a JD.")
    r.add_argument("jd", help="JD name, e.g. frontend_react_jd")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_rank)

    g = sub.add_parser("gaps", help="Skill-gap analysis for a resume/JD pair.")
    g.add_argument("resume", help="Resume name, e.g. frontend_vue")
    g.add_argument("jd", help="JD name, e.g. frontend_react_jd")
    g.add_argument("--target", type=float, default=90.0)
    g.set_defaults(func=cmd_gaps)

    d = sub.add_parser("demo", help="Run the end-to-end demo.")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        missing = exc.filename or str(exc)
        print(
            f"Error: could not find {missing}. Use a bundled name like "
            f"'backend_java' or 'frontend_react_jd' (no .md suffix).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
