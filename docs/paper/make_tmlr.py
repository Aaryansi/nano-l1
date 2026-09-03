"""generate the TMLR submission from main.tex, so there is one source of truth.

TMLR requires its own single-column stylefile and rejects non-anonymous
submissions without review. main.tex is a self-contained two-column format,
which suits arXiv and the repository copy. maintaining two hand-edited sources
is how this project has already shipped three documents that drifted into
contradicting each other, so the submission is generated instead.

what this rewrites, and nothing else:
  - the preamble, to tmlr.sty. anonymity is the stylefile's job: without the
    [accepted] or [preprint] option it hides the author block itself, so the
    \\anon switch in main.tex is dropped rather than carried over.
  - the two-column abstract wrapper, which has no meaning single-column.
  - starred floats. figure* and table* span both columns in a two-column
    layout and are a syntax error in a single-column one.
  - \\columnwidth, which is \\textwidth here.
  - the bibliography style, to tmlr.

usage:
    python make_tmlr.py          # writes tmlr/main.tex and copies assets
    cd tmlr && tectonic main.tex
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tmlr"

PREAMBLE = r"""% GENERATED FILE, DO NOT EDIT. Produced by make_tmlr.py from main.tex.
% Edit main.tex and regenerate, or the two will drift.
%
% Build: tectonic main.tex
% Anonymity is handled by tmlr.sty: without [accepted] or [preprint] it hides
% the author block. Do not add the option before acceptance.

\documentclass[10pt]{article}
\usepackage{tmlr}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{microtype}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{url}

\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlength{\tabcolsep}{4pt}

\newcommand{\vfn}{v}

\title{No Free Null: Reference Distributions for Explaining\\
Reinforcement Learning Agents}

% tmlr.sty hides this until the [accepted] option is set.
\author{\name Aaryan Singh \email singha9@rose-hulman.edu \\
      \addr Independent Researcher}

\def\month{MM}
\def\year{2026}
\def\openreview{\url{https://openreview.net/forum?id=XXXX}}

\begin{document}
\maketitle

\begin{abstract}
"""


def convert(src: str) -> str:
    # body starts after the two-column abstract wrapper; the abstract text is
    # the quote block inside it.
    m = re.search(r"\\textbf\{Abstract\.\}\s*(.*?)\s*\\end\{quote\}", src, re.S)
    if not m:
        raise SystemExit("could not locate the abstract in main.tex")
    abstract = m.group(1).strip()

    body_start = src.index("\\section{Introduction}")
    body = src[body_start:]

    body = body.replace("\\begin{figure*}", "\\begin{figure}")
    body = body.replace("\\end{figure*}", "\\end{figure}")
    body = body.replace("\\begin{table*}", "\\begin{table}")
    body = body.replace("\\end{table*}", "\\end{table}")
    body = body.replace("\\columnwidth", "\\textwidth")
    body = body.replace("\\bibliographystyle{plainnat}", "\\bibliographystyle{tmlr}")

    # main.tex guards the repository URL behind \ifanon. tmlr.sty owns
    # anonymity here and \ifanon is undefined, so collapse every such guard to
    # its anonymous branch. a submission that leaked the URL would leak the
    # author's username with it.
    body = re.sub(r"\\ifanon(.*?)\\else.*?\\fi", lambda m: m.group(1), body, flags=re.S)
    if "\\ifanon" in body or "github.com/Aaryansi" in body:
        raise SystemExit("anonymity guard failed: identifying content survived")

    return PREAMBLE + abstract + "\n\\end{abstract}\n\n" + body


def main() -> None:
    src = (HERE / "main.tex").read_text()
    OUT.mkdir(exist_ok=True)
    (OUT / "main.tex").write_text(convert(src))

    shutil.copy(HERE / "references.bib", OUT / "references.bib")
    if (OUT / "figures").exists():
        shutil.rmtree(OUT / "figures")
    shutil.copytree(HERE / "figures", OUT / "figures")

    missing = [f for f in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty")
               if not (OUT / f).exists()]
    print(f"wrote {OUT / 'main.tex'}")
    if missing:
        print("\nstill needed from https://github.com/JmlrOrg/tmlr-style-file:")
        for f in missing:
            print(f"  {f}")


if __name__ == "__main__":
    main()
