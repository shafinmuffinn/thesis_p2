"""Static verification of the thesis LaTeX sources.

Checks that every \\ref{...} and \\cite{...} resolves against the
\\label{...} entries and the bibliography keys, without needing a
local TeX installation. Catches the same kinds of problems that a
pdflatex + biber + pdflatex pass would flag as ?? references or
undefined citations.

Reports:
    - undefined \\ref{...} targets        (would render as ?? in PDF)
    - undefined \\cite{...} keys          (would render as [?] in PDF)
    - duplicate \\label{...} definitions  (warning)
    - duplicate bib keys                  (warning)
    - unused \\label{...} entries         (informational)
    - unused bib keys                     (informational)
    - cross-reference summary
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# `python verify_latex.py` checks the P3 report; pass `LateX` to check the P2 one.
ROOT = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "LateX_P3")
TEX_GLOBS = ["main.tex", "chapters/*.tex", "core/*.tex", "appendix/*.tex"]
BIB_FILE = ROOT / "bibliography" / "references.bib"


def collect_tex_files() -> list[Path]:
    files: list[Path] = []
    for pat in TEX_GLOBS:
        files.extend(sorted(ROOT.glob(pat)))
    return files


def parse_tex(files: list[Path]) -> tuple[dict, dict, dict]:
    """Return three dicts: labels, refs, cites, each mapping
    name -> list of (file, line) tuples."""
    label_re = re.compile(r"\\label\s*\{([^}]+)\}")
    ref_re = re.compile(r"\\(?:ref|eqref|autoref|pageref|nameref)\s*\{([^}]+)\}")
    cite_re = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")

    labels: dict[str, list] = defaultdict(list)
    refs: dict[str, list] = defaultdict(list)
    cites: dict[str, list] = defaultdict(list)

    for f in files:
        try:
            text = f.read_text()
        except Exception as e:
            print(f"WARN: could not read {f}: {e}")
            continue
        for ln_no, line in enumerate(text.splitlines(), start=1):
            for m in label_re.finditer(line):
                labels[m.group(1)].append((f, ln_no))
            for m in ref_re.finditer(line):
                refs[m.group(1)].append((f, ln_no))
            for m in cite_re.finditer(line):
                # \cite can take comma-separated keys
                for key in m.group(1).split(","):
                    key = key.strip()
                    if key:
                        cites[key].append((f, ln_no))
    return labels, refs, cites


def parse_bib(bib: Path) -> dict[str, int]:
    """Return dict mapping bib key -> line number of @entry."""
    if not bib.exists():
        return {}
    key_re = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
    keys: dict[str, int] = {}
    text = bib.read_text()
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = key_re.match(line)
        if m:
            k = m.group(1)
            if k not in keys:
                keys[k] = line_no
            else:
                # duplicate
                keys.setdefault(f"__DUP:{k}", line_no)
    return keys


def relp(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT.parent))
    except ValueError:
        return str(p)


def main() -> int:
    files = collect_tex_files()
    if not files:
        print(f"ERROR: no .tex files found under {ROOT}")
        return 2

    print(f"Scanning {len(files)} .tex files under {ROOT}")
    print(f"Bibliography: {BIB_FILE}")
    print()

    labels, refs, cites = parse_tex(files)
    bib_keys = parse_bib(BIB_FILE)

    # --- Undefined refs (would render as ?? in PDF) ---
    undefined_refs = [r for r in refs if r not in labels]
    print(f"=== \\ref / \\autoref / \\eqref targets ({len(refs)} unique used) ===")
    if undefined_refs:
        print(f"  FAIL: {len(undefined_refs)} undefined target(s):")
        for r in sorted(undefined_refs):
            for f, ln in refs[r]:
                print(f"    \\ref{{{r}}}  used at {relp(f)}:{ln}")
    else:
        print("  OK: every \\ref{...} resolves to a \\label{...}")
    print()

    # --- Undefined cites (would render as [?] in PDF) ---
    real_bib_keys = {k for k in bib_keys if not k.startswith("__DUP:")}
    undefined_cites = [c for c in cites if c not in real_bib_keys]
    print(f"=== \\cite{{...}} keys ({len(cites)} unique used) ===")
    if undefined_cites:
        print(f"  FAIL: {len(undefined_cites)} undefined citation key(s):")
        for c in sorted(undefined_cites):
            for f, ln in cites[c]:
                print(f"    \\cite{{{c}}}  used at {relp(f)}:{ln}")
    else:
        print("  OK: every \\cite{...} resolves to a bib entry")
    print()

    # --- Duplicate labels ---
    dup_labels = {k: v for k, v in labels.items() if len(v) > 1}
    if dup_labels:
        print(f"=== Duplicate \\label{{...}} (warning) ===")
        for k, locs in sorted(dup_labels.items()):
            print(f"  {k}:")
            for f, ln in locs:
                print(f"    {relp(f)}:{ln}")
        print()

    # --- Duplicate bib keys ---
    dup_bib = [k.removeprefix("__DUP:") for k in bib_keys
               if k.startswith("__DUP:")]
    if dup_bib:
        print("=== Duplicate bib keys (warning) ===")
        for k in dup_bib:
            print(f"  {k}")
        print()

    # --- Summary ---
    print("=== Summary ===")
    print(f"  tex files scanned   : {len(files)}")
    print(f"  \\label{{...}}       : {len(labels)} unique")
    print(f"  \\ref{{...}}         : {len(refs)} unique  "
          f"({sum(len(v) for v in refs.values())} occurrences)")
    print(f"  \\cite{{...}}        : {len(cites)} unique  "
          f"({sum(len(v) for v in cites.values())} occurrences)")
    print(f"  bib entries         : {len(real_bib_keys)} unique")
    print(f"  bib entries unused  : {len(real_bib_keys - set(cites))}")
    print(f"  duplicate labels    : {len(dup_labels)}")
    print(f"  duplicate bib keys  : {len(dup_bib)}")
    # \pend{...} marks a value not yet traced to a results CSV (main.tex).
    pend_re = re.compile(r"\\pend\{")
    pending = [(f, n) for f in files if f.name != "main.tex"
               for n, line in enumerate(f.read_text().splitlines(), start=1)
               if pend_re.search(line)]
    print(f"  \\pend{{...}} markers : {len(pending)}")
    for f, n in pending:
        print(f"      {relp(f)}:{n}")
    print()
    if undefined_refs or undefined_cites:
        print("RESULT: FAIL  -- the issues above would render as ?? / [?] in PDF.")
        return 1
    if pending:
        print("RESULT: PASS (refs/cites) -- but resolve every \\pend{} before submitting.")
        return 0
    print("RESULT: PASS  -- no broken references or citations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
