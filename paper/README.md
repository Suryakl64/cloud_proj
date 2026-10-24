# IEEE Research Paper: ML Inference Crossover Analysis

This directory contains the complete IEEE-formatted research paper based on the project:
**"Cost and Latency Crossover Analysis of Machine Learning Inference Deployment Modes: An Empirical Study on AWS EC2, Lambda, and SageMaker"**

---

## Directory Contents

* **`main.tex`**: Complete IEEE-formatted LaTeX source code using `IEEEtran.cls` with all mathematical formulas, tables, figure references, and citations.
* **`references.bib`**: BibTeX bibliography file with academic citations (USENIX ATC, ASPLOS, NSDI, IEEE ICDCS, CVPR).
* **`paper_content.md`**: Complete human-readable Markdown version of the paper. Useful if you want to copy-paste sections directly into Microsoft Word or Google Docs.
* **`figures/`**: High-resolution 300 DPI publication figures referenced by the LaTeX paper (`architecture_simple.png`, `live_comparison.png`, `fig1_cost_crossover.png`).

---

## How to Compile & Generate Your PDF

### Method 1: Using Overleaf (Recommended - Fastest & Zero Setup)
1. Go to [Overleaf](https://www.overleaf.com/) and log in (or create a free account).
2. Click **New Project** -> **Upload Project**.
3. Select this `paper/` folder (or zip the `paper/` folder and upload `paper.zip`).
4. Overleaf will automatically open `main.tex` and load the official `IEEEtran` compiler.
5. Click **Recompile** -> Download your publication-ready IEEE PDF!

### Method 2: Using Local LaTeX (TeX Live / MiKTeX)
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Method 3: Using Microsoft Word (IEEE Template)
1. Download the official [IEEE MS Word Template](https://www.ieee.org/conferences/publishing/templates.html) (`.docx`).
2. Open [`paper_content.md`](paper_content.md).
3. Copy-paste each section (Abstract, Intro, Related Work, Architecture, Math, Results, Conclusion) into the respective template sections.
4. Insert the 300 DPI figures from the `paper/figures/` directory.
