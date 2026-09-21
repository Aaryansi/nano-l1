"""build every venue target from main.tex, so no two versions can drift.

this project has already shipped three documents that drifted into
contradicting the paper (PAPER.md, main.md, REPORT.md). the fix is not
discipline, it is refusing to keep a second editable copy.

targets:
  arxiv     the full argument, two-column, self-contained. also what the xAI
            World Conference full-article track (14-24pp) wants.
  core      8 pages of main body with everything else after the references.
            8pp is the binding constraint across venues: ICML auto-rejects
            over 8, ICLR allows 9, NeurIPS 9-10, TMLR Regular 12. writing to
            8 means one file is submittable to all of them.
  tmlr      the core body in TMLR's stylefile, for a Regular resubmission.

the appendix split is a claim about what a reviewer must read to agree with
the paper, not about what is worth keeping. every relocated section leaves a
sentence in the body saying what it establishes and where it went.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# sections whose evidence supports the argument but is not required to follow
# it. ordered as they should appear in the appendix.
RELOCATE = [
    "Reproducibility",
    "Reproduction from scratch",
    "A positive control, under the blinded reference",
    "How much edge is required",
    "Estimation guarantees do not substitute",
    "The canonical check, and what it answers instead",
    "What survives the choice of null",
    "What did not survive",
]

# what the body says in place of each relocated section. these are claims, not
# signposts: a reader who never opens the appendix should still learn the
# result. keyed by the section title above.
STUBS = {
    "Reproducibility": r"""
\paragraph{Reproducibility.} All results derive from public endpoints with no
API keys and run on CPU. Hyperparameters were fixed a priori on synthetic
sweeps with analytically known answers. The full pipeline is one command; 312
tests cover the environment and the attribution implementations, and every
numerical claim in this paper is asserted against its source artifact by a
script that runs as the pipeline's final stage. Budgets, hardware, the exact
commands, and a from-scratch rebuild on a different sample of the moving
market window are in Appendix~\ref{app:repro}.
""",
    "A positive control, under the blinded reference": r"""
A positive control on the same real episodes, holding corpus, features,
normalizer and split fixed and varying only the objective, shows the test
separates tasks and not datasets: an agent scored for \emph{calling} the
settlement fires at $z=+3.39$ where the trading agent does not. It uses the
blinded reference that Section~\ref{sec:matched} goes on to reject, so we
report it as a controlled comparison and not as evidence the reference is
sound (Appendix~\ref{app:positive}).
""",
    "How much edge is required": r"""
Sweeping planted signal strength puts the detection threshold near $+3.45$ per
episode of measured edge, an order of magnitude above anything the market agent
achieves (Appendix~\ref{app:edge}).
""",
    "Estimation guarantees do not substitute": r"""
Monte Carlo certification of a top-$k$ ranking~\citep{goldwasser2024significance}
is orthogonal to this: the market agent's top-5 ranking is fully certified and
its span is still indistinguishable from an explanation of nothing
(Appendix~\ref{app:certified}).
""",
    "The canonical check, and what it answers instead": r"""
Parameter randomisation is also used a second way, as
\citet{adebayo2018sanity} propose it: degrade one explanation and ask whether
it moves. We ran that too. It clears the empty explanation on both explanatory
targets we tried ($\rho=0.38$ on behaviour, $\rho=-0.02$ on outcomes), so
passing it is not evidence that there is anything to explain. What it flags
depends on the target, which is a second instance of this paper's thesis
(Appendix~\ref{app:canonical}).
""",
    "What survives the choice of null": r"""
\paragraph{What survives, and what does not.}
Holding the null fixed, the verdict is robust to choices that move the ranking
a great deal. It is unchanged across attribution family, across
credit-assignment scheme, and under a $k$-nearest-neighbour conditional masker
that substantially reduces off-manifold states, while the per-feature ranking
moves under all three, to the point of disagreeing on which feature matters
most. The verdict is \emph{not} robust to the null construction, which is the
result of this section. Separately, an auxiliary training penalty drives a
feature's attribution from $39.9\%$ to $3.2\%$ of total mass at no measurable
cost in return, so attribution is not identified by task performance.
Bootstrapping the reference leaves every verdict unchanged in $100\%$ of
replicates except the two blinded-real comparisons, at $64\%$ and $65\%$
(Appendix~\ref{app:survives}).
""",
    "What did not survive": r"""
Seven predictions this project made were rejected on its own evidence,
including our own expectation that instability would be the tell for an empty
explanation and that the canonical check would agree with the null test. None
of the surviving results depends on any of them
(Appendix~\ref{app:rejected}).
""",
}


BLOCK_STUBS = {
    "The stratified permutation": r"""
The obvious repair is to stratify, shuffling outcomes only among contracts the
market priced alike so each price bucket keeps its base rate and the arbitrage
never appears. It works, and it is vacuous here: from eight buckets on the
calibration error returns to the real corpus's $0.007$, but the relabelling then
moves only $5$--$6\%$ of outcomes and the permuted labels stay correlated at
$0.88$--$0.90$ with the true ones at every resolution. The terminal price
correlates $0.950$ with the outcome and $88.7\%$ of episodes end within $0.1$ of
a bound, so information beyond the price lives in a small minority of episodes
and a relabelling that removes only that cannot differ much from the real data.
The construction removes the right thing; there is almost none of it here to
remove. Unlike the other three this failure is contingent on the domain
(Appendix~\ref{app:secondary}).
""",
}


@dataclass
class Block:
    level: int          # 1 = section, 2 = subsection
    title: str
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines)


def parse(src: str) -> tuple[str, list[Block]]:
    """split the body into section blocks, keeping the preamble intact."""
    lines = src.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("\\section{Introduction}"))
    preamble = "\n".join(lines[:start])

    blocks: list[Block] = []
    for line in lines[start:]:
        m = re.match(r"\\(sub)?section\{(.+)\}\s*$", line)
        if m:
            blocks.append(Block(2 if m.group(1) else 1, m.group(2)))
        if blocks:
            blocks[-1].lines.append(line)
    return preamble, blocks


def split_body(blocks: list[Block]) -> tuple[list[Block], list[Block]]:
    """partition into body and appendix, keeping subsections with their parent."""
    body, appendix = [], []
    # relocation is scoped: naming a section moves it and its subsections,
    # naming a subsection moves only that subsection. the earlier version was
    # sticky and swept 6.3 to 6.5 into the appendix, which are the result.
    active: int | None = None
    for b in blocks:
        if b.title in RELOCATE:
            active = b.level
        elif active is not None and b.level <= active:
            active = None
        (appendix if active is not None else body).append(b)
    return body, appendix


LABELS = {
    "Reproducibility": "app:repro",
    "A positive control, under the blinded reference": "app:positive",
    "How much edge is required": "app:edge",
    "Estimation guarantees do not substitute": "app:certified",
    "The canonical check, and what it answers instead": "app:canonical",
    "What survives the choice of null": "app:survives",
    "What did not survive": "app:rejected",
}


def emit(blocks: list[Block]) -> tuple[str, str]:
    """render body and appendix, inserting each stub where its section was.

    a stub is attached to the last body block written before the relocation,
    so the reader meets the claim at the point in the argument where it was
    originally made rather than discovering a gap.
    """
    body: list[str] = []
    appendix: list[str] = ["\n\\appendix\n"]
    active: int | None = None

    for b in blocks:
        relocate = b.title in RELOCATE
        if relocate:
            active = b.level
        elif active is not None and b.level <= active:
            active = None

        if active is None:
            body.append(b.text())
            continue

        if relocate and b.title in STUBS:
            body.append(STUBS[b.title].strip() + "\n")

        t = b.text()
        if b.title in LABELS:
            head = f"\\{'sub' if b.level == 2 else ''}section{{{b.title}}}"
            t = t.replace(head, head + f"\n\\label{{{LABELS[b.title]}}}", 1)
        appendix.append(t)

    return "\n".join(body), "\n".join(appendix)


def main() -> None:
    src = (HERE / "main.tex").read_text()
    preamble, blocks = parse(src)
    body, appendix = emit(blocks)

    out = HERE / "core"
    out.mkdir(exist_ok=True)

    # figures marked %%% MOVE-FIGURE support the argument without carrying it.
    # they follow their section's evidence into the appendix, which is worth
    # roughly 0.4 of a page each in single column.
    moved_figs: list[str] = []

    def _lift(text: str) -> str:
        pat = re.compile(r"%%% MOVE-FIGURE\n(\\begin\{figure\*?\}.*?\\end\{figure\*?\})\n",
                         re.S)
        def take(m):
            moved_figs.append(m.group(1))
            return ""
        return pat.sub(take, text)

    body = _lift(body)

    # whole sub-analyses marked %%% MOVE-BLOCK ... %%% MOVE-END are secondary
    # evidence within a section that stays. they follow the figures into the
    # appendix and leave a stub in their place.
    moved_blocks: list[str] = []

    def _lift_block(text: str) -> str:
        pat = re.compile(r"%%% MOVE-BLOCK: ([^\n]+)\n(.*?)%%% MOVE-END\n", re.S)
        def take(m):
            moved_blocks.append((m.group(1).strip(), m.group(2)))
            return BLOCK_STUBS.get(m.group(1).strip(), "")
        return pat.sub(take, text)

    body = _lift_block(body)

    doc = body + "\n" + appendix

    # single-column transforms. starred floats span both columns and are a
    # syntax error here; \columnwidth is \textwidth; tmlr.sty owns anonymity,
    # so \ifanon is undefined and every guard collapses to its anon branch.
    doc = re.sub(r"\\(begin|end)\{(figure|table)\*\}", r"\\\1{\2}", doc)
    doc = doc.replace("\\columnwidth", "\\textwidth")
    doc = re.sub(r"\\ifanon(.*?)\\else.*?\\fi", lambda m: m.group(1), doc, flags=re.S)
    if "\\ifanon" in doc or "github.com/Aaryansi" in doc:
        raise SystemExit("anonymity guard failed: identifying content survived")
    # the bibliography must precede the appendix: main content is everything
    # before the references, which is the quantity every page limit counts.
    doc = doc.replace("\\bibliographystyle{plainnat}\n\\bibliography{references}", "")
    doc = doc.replace("\\end{document}", "")
    if moved_blocks:
        parts = [f"\\subsection{{{t}}}\n{c}" for t, c in moved_blocks]
        doc = doc.replace("\n\\appendix\n",
                          "\n\\appendix\n\n\\section{Secondary analyses}\n"
                          "\\label{app:secondary}\n\n" + "\n\n".join(parts) + "\n", 1)
    if moved_figs:
        figs = "\n\n".join(moved_figs)
        doc = doc.replace("\n\\appendix\n",
                          "\n\\appendix\n\n\\section{Supporting figures}\n"
                          "\\label{app:figures}\n\n" + figs + "\n", 1)
    doc = doc.replace("\n\\appendix\n",
                      "\n\\bibliographystyle{tmlr}\n\\bibliography{references}\n\n\\appendix\n")
    doc += "\n\\end{document}\n"

    m = re.search(r"\\textbf\{Abstract\.\}\s*(.*?)\s*\\end\{quote\}", src, re.S)
    if not m:
        raise SystemExit("could not find the abstract in main.tex")
    abstract = m.group(1).strip()
    (out / "main.tex").write_text(
        tmlr_preamble().replace("ABSTRACT_PLACEHOLDER", abstract) + doc)

    for asset in ("references.bib",):
        shutil.copy(HERE / asset, out / asset)
    if (out / "figures").exists():
        shutil.rmtree(out / "figures")
    shutil.copytree(HERE / "figures", out / "figures")
    for sty in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty"):
        src_sty = HERE / "tmlr" / sty
        if src_sty.exists():
            shutil.copy(src_sty, out / sty)

    print(f"wrote {out / 'main.tex'}")


def tmlr_preamble() -> str:
    return r"""% GENERATED by build.py from main.tex. Do not edit; edit main.tex.
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
\author{\name Aaryan Singh \email singha9@rose-hulman.edu \\
      \addr Independent Researcher}
\def\month{MM}
\def\year{2026}
\def\openreview{\url{https://openreview.net/forum}}
\begin{document}
\maketitle
\begin{abstract}
ABSTRACT_PLACEHOLDER
\end{abstract}

"""


if __name__ == "__main__":
    main()
