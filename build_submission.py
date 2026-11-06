"""Assemble the Phase 3 submission ZIP, behind guards.

    python build_submission.py            # check every guard, then build the ZIP
    python build_submission.py --check    # check every guard, write nothing

Produces Team10_IT5006_Ecommerce_Analytics_AY2627Sem1.zip in the project root
with exactly five members: the final report PDF, the presentation slides (PPTX
and PDF, byte-identical to the deck locked on 4 November 2026), GITHUB_LINK.txt
(repository, application and the commit submitted) and a copy of the README so
the archive is self-describing. The ZIP itself is not tracked in git.

Every guard runs before anything is written. If any fails, the script lists
the failures and exits with status 1 without writing or overwriting an archive:

  1. both slide files match the SHA-256 of the deck locked on 4 November;
  2. docs/ holds the same report PDF as report/ (byte-identical);
  3. the body, from the page of the sec:executive label to the page of the
     sec:bodyend label in report/Team10_Phase3_IT5006_AY2627Sem1.aux, is at
     most 20 pages; the same .aux supplies the appendix and table numbers that
     GITHUB_LINK.txt cites, so the note always matches the PDF;
  4. the working tree is clean (`git status --porcelain` prints nothing) and
     HEAD is origin/main, so the commit named in GITHUB_LINK.txt is the one on
     GitHub. When the network allows, GitHub's main is also read with
     `git ls-remote` (read-only) and must be the same commit.

The archive is written to a temporary file next to the target, re-read and
checked member by member against the sources, and only then moved into place.
The SHA-256 of every archived file is printed.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEAM = "Team10"
ZIP = ROOT / f"{TEAM}_IT5006_Ecommerce_Analytics_AY2627Sem1.zip"
REPO = "https://github.com/gengyuzccc-cell/Team10_IT5006_Ecommerce_Analytics_AY2627Sem1"
APP = "https://team10-it5006-olist.streamlit.app"

REPORT = f"{TEAM}_Phase3_IT5006_AY2627Sem1"
DOCS_PDF = ROOT / "docs" / f"{REPORT}.pdf"
BUILT_PDF = ROOT / "report" / f"{REPORT}.pdf"
AUX = ROOT / "report" / f"{REPORT}.aux"
MAX_BODY_PAGES = 20

# The deck locked on 4 November 2026. It is never rebuilt or re-exported.
LOCKED_SLIDES = {
    f"{TEAM}_Phase3_slides.pptx": "189cd5afafb7d2dd77b5ff8fc9c759fda0b197a2c5a2557dd43ffd736536cfe8",
    f"{TEAM}_Phase3_slides.pdf": "f0f159b4fd08f327a1ff5f9c5f2f175c23f125e1467b831366e5f8dde2f77395",
}

# Name inside the ZIP -> file on disk.
FILES = {
    f"{REPORT}.pdf": DOCS_PDF,
    f"{TEAM}_Phase3_slides.pptx": ROOT / "docs" / f"{TEAM}_Phase3_slides.pptx",
    f"{TEAM}_Phase3_slides.pdf": ROOT / "docs" / f"{TEAM}_Phase3_slides.pdf",
    "README.md": ROOT / "README.md",
}
LINK_NAME = "GITHUB_LINK.txt"
MEMBERS = {*FILES, LINK_NAME}

_NEWLABEL = re.compile(r"\\newlabel\{([^{}]*)\}\{\{([^{}]*)\}\{([^{}]*)\}")


def sha256(data_or_path: bytes | Path) -> str:
    if isinstance(data_or_path, bytes):
        return hashlib.sha256(data_or_path).hexdigest()
    digest = hashlib.sha256()
    with open(data_or_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, timeout: float | None = None) -> str:
    """Run a read-only git command in the project root and return its stdout."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.rstrip()   # keep the leading space of porcelain status lines


def aux_labels(path: Path) -> dict[str, tuple[str, str]]:
    """label -> (number, page), from the report's \\newlabel lines."""
    labels = {}
    for line in path.read_text("utf-8", errors="replace").splitlines():
        m = _NEWLABEL.match(line)
        if m:
            labels[m.group(1)] = (m.group(2), m.group(3))
    return labels


def check_guards() -> tuple[list[str], dict]:
    """Every guard, without writing anything. Returns (failures, facts)."""
    failures: list[str] = []
    facts: dict = {}

    sources = [*FILES.values(), BUILT_PDF, AUX]
    missing = [p.relative_to(ROOT).as_posix() for p in sources if not p.exists()]
    if missing:
        failures.append("missing: " + ", ".join(missing)
                        + " (build the report in report/, then copy its PDF to docs/)")

    # 1. The slides are the deck locked on 4 November.
    for name, locked in LOCKED_SLIDES.items():
        path = FILES[name]
        if path.exists():
            got = sha256(path)
            if got != locked:
                failures.append(f"docs/{name} is not the deck locked on 4 November "
                                f"(sha256 {got[:12]}..., locked {locked[:12]}...)")

    # 2. The report in docs/ is the report just built in report/.
    if DOCS_PDF.exists() and BUILT_PDF.exists() and sha256(DOCS_PDF) != sha256(BUILT_PDF):
        failures.append(f"docs/{REPORT}.pdf differs from report/{REPORT}.pdf: "
                        "copy the freshly built report to docs/")

    # 3. The body fits the brief; the note's appendix and table numbers exist.
    if AUX.exists():
        labels = aux_labels(AUX)
        try:
            first, last = int(labels["sec:executive"][1]), int(labels["sec:bodyend"][1])
            facts["body"] = (first, last, last - first + 1)
            if last - first + 1 > MAX_BODY_PAGES:
                failures.append(f"the body runs {last - first + 1} pages (p.{first}-{last}); "
                                f"the brief allows {MAX_BODY_PAGES}")
        except KeyError as exc:
            failures.append(f"{AUX.name} has no {exc} label: cannot measure the body")
        except ValueError:
            failures.append(f"{AUX.name}: the page of sec:executive or sec:bodyend is not a number")
        cited = {}
        for key, label in (("feedback", "app:feedback"), ("table", "tab:feedback3"),
                           ("provenance", "app:provenance")):
            if labels.get(label, ("", ""))[0]:
                cited[key] = labels[label][0]
            else:
                failures.append(f"{AUX.name} has no {label} label: build the report revised "
                                "after the presentation (GITHUB_LINK.txt cites it)")
        facts["cited"] = cited
        if BUILT_PDF.exists():
            gap = abs(AUX.stat().st_mtime - BUILT_PDF.stat().st_mtime) / 60
            facts["aux_pdf_gap_min"] = gap

    # 4. The tree is committed and pushed, so the commit named is on GitHub.
    try:
        dirty = git("status", "--porcelain").splitlines()
        if dirty:
            shown = "\n".join("      " + d for d in dirty[:12])
            more = f"\n      ... and {len(dirty) - 12} more" if len(dirty) > 12 else ""
            failures.append(f"the working tree is not clean ({len(dirty)} paths); commit and "
                            f"push first:\n{shown}{more}")
        head = git("rev-parse", "HEAD")
        facts["head"] = head
        try:
            upstream = git("rev-parse", "--verify", "--quiet", "origin/main")
        except RuntimeError:
            upstream = ""
        if upstream != head:
            failures.append(f"HEAD {head[:12]} is not origin/main "
                            f"{upstream[:12] if upstream else '(no such ref)'}: push first")
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        failures.append(f"git could not be read: {exc}")
    return failures, facts


def check_github(head: str) -> str | None:
    """GitHub's main, read with `git ls-remote` (read-only). None if unreachable."""
    try:
        line = git("ls-remote", "origin", "refs/heads/main", timeout=30)
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None
    return line.split()[0] if line else ""


def link_text(head: str, cited: dict, digests: dict[str, str]) -> str:
    note = (f"The report was revised after the 5 November presentation; the changes are "
            f"listed in its Appendix {cited['feedback']} (Table {cited['table']}) and Appendix "
            f"{cited['provenance']}. The slides are byte-identical to the deck locked on "
            f"4 November.")
    contents = [
        (f"{REPORT}.pdf", "final report (application and GitHub links on the cover)"),
        (f"{TEAM}_Phase3_slides.pptx / .pdf", "presentation slides, as locked on 4 November"),
        ("README.md", "repository overview and setup instructions"),
        (LINK_NAME, "this file"),
    ]
    width = max(len(name) for name, _ in contents) + 3
    listing = "\n".join(f"  {name.ljust(width)}{what}" for name, what in contents)
    hashes = "\n".join(f"  {digests[name]}  {name}" for name in FILES)
    return (
        f"IT5006 Team 10 - Phase 3 submission\n"
        f"\n"
        f"GitHub repository : {REPO}\n"
        f"Deployed app      : {APP}\n"
        f"Repository version submitted: {head}\n"
        f"\n"
        f"{textwrap.fill(note, width=78, break_on_hyphens=False)}\n"
        f"\n"
        f"Contents of this archive\n"
        f"{listing}\n"
        f"\n"
        f"SHA-256 of the other files in this archive\n"
        f"{hashes}\n"
        f"\n"
        f"Team 10: Pan Pinyou, Li Yudan, Zhu Gengyu\n"
    )


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    print("Checking the submission guards ...")
    failures, facts = check_guards()

    if "body" in facts:
        first, last, pages = facts["body"]
        print(f"  body: p.{first}-{last}, {pages} pages (limit {MAX_BODY_PAGES})")
    if facts.get("aux_pdf_gap_min", 0) > 10:
        print(f"  warning: {AUX.name} and the report PDF were written "
              f"{facts['aux_pdf_gap_min']:.0f} minutes apart; if the PDF did not come from "
              "the same pdflatex run, rebuild before trusting the page count")
    if not failures:
        remote = check_github(facts["head"])
        if remote is None:
            print("  warning: GitHub could not be reached; relying on the local origin/main ref")
        elif remote != facts["head"]:
            failures.append(f"GitHub's main is {remote[:12] or '(missing)'}, not HEAD "
                            f"{facts['head'][:12]}: fetch, reconcile and push first")
        else:
            print(f"  GitHub main = HEAD = {facts['head'][:12]}")

    if failures:
        print(f"\nREFUSED: {len(failures)} guard(s) failed; nothing was written.")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("  every guard passed: slides locked, docs/ PDF = report/ PDF, body within the "
          "limit, tree clean and pushed")
    if check_only:
        return 0

    digests = {name: sha256(path) for name, path in FILES.items()}
    text = link_text(facts["head"], facts["cited"], digests)
    digests[LINK_NAME] = sha256(text.encode("utf-8"))

    tmp = ZIP.with_name(ZIP.stem + ".partial.zip")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, path in FILES.items():
                zf.write(path, name)
            zf.writestr(LINK_NAME, text)
        with zipfile.ZipFile(tmp) as zf:
            names = set(zf.namelist())
            if names != MEMBERS or zf.testzip() is not None:
                raise RuntimeError(f"archive check failed: members {sorted(names)}")
            changed = [n for n in MEMBERS if sha256(zf.read(n)) != digests[n]]
            if changed:
                raise RuntimeError(f"archived bytes differ from the sources: {changed}")
        os.replace(tmp, ZIP)
    except Exception as exc:  # noqa: BLE001 - never leave a half-written archive
        print(f"\nREFUSED: {exc}; the previous archive, if any, is untouched.")
        return 1
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"\nwrote {ZIP.name} ({ZIP.stat().st_size / 1e6:.1f} MB), commit {facts['head']}")
    for name in sorted(MEMBERS):
        print(f"  {digests[name]}  {name}")
    print(f"  {sha256(ZIP)}  {ZIP.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
