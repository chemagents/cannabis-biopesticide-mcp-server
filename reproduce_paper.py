#!/usr/bin/env python
"""Reproduce the headline results & conclusions of the Cannabis biopesticide paper and print them.

    python reproduce_paper.py     # recomputes docking stats, RMT filter, QSAR, candidates
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from server.canpest_server import reproduce_all, reproduce_claims  # noqa: E402


def main() -> None:
    ra = getattr(reproduce_all, "fn", reproduce_all)()["answer"]
    print("HEADLINE NUMBERS vs the paper")
    print("=" * 60)
    print(ra["summary"])
    for c in ra["checks"]:
        print(f"  [{'PASS' if c['match'] else 'FAIL'}] {c['metric']}: {c['reproduced']} vs {c['paper']}")

    rc = getattr(reproduce_claims, "fn", reproduce_claims)()["answer"]
    print("\nCONCLUSIONS")
    print("=" * 60)
    print(f"reproduced {rc['reproduced']}/{rc['total']} claims")
    for c in rc["claims"]:
        print(f"  [{c['id']}] {'OK' if c['reproduced'] else 'NO'}  {c['reproduced_statement'][:95]}")


if __name__ == "__main__":
    main()
