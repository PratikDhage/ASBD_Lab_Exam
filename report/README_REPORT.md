# LaTeX Report — Compilation Guide

## Directory Structure

```
report/
├── main.tex          ← Complete LaTeX source
├── images/           ← Copy all PNG files here (see checklist below)
└── README_REPORT.md  ← This file
```

## How to Compile

### Option 1: pdflatex (recommended)
```bash
cd report
pdflatex main.tex
pdflatex main.tex   # Run twice for TOC and cross-references
```

### Option 2: latexmk (automated)
```bash
cd report
latexmk -pdf main.tex
```

### Option 3: Overleaf
1. Upload `main.tex` and all images from `images/` folder.
2. Compile with pdfLaTeX.

## Required LaTeX Packages

The following packages are used (all are standard in TeX Live / MiKTeX Full):
- `geometry`, `graphicx`, `booktabs`, `longtable`, `tabularx`, `float`
- `caption`, `subcaption`, `amsmath`, `amssymb`, `enumitem`
- `xcolor`, `hyperref`, `tcolorbox` (with `most` option)
- `fancyhdr`, `titlesec`, `setspace`, `parskip`, `pdflscape`, `makecell`
- `array`, `multirow`, `lmodern`, `inputenc`, `fontenc`

If you are missing packages, install TeX Live Full or MiKTeX Full.

## Editable Placeholders

Search the `.tex` file for these placeholders and replace them with your actual values:

| Placeholder | Location | Replace With |
|---|---|---|
| `[COURSE_CODE]` | Cover page | Your course code (e.g., CS6024) |
| `[ROLL_NUMBER]` | Cover page | Your roll number |
| `[INSTRUCTOR_NAME]` | Cover page | Instructor's name |
| `[GITHUB_REPO_URL]` | Cover page | Your GitHub repository URL |

## Graceful Image Handling

The report uses `\IfFileExists` via the `\safeimg` macro. If an image file is missing from the `images/` directory, a placeholder box will appear instead. The document will compile successfully even with no images.

## Expected Page Count

With all images included: **24–28 pages** (depending on image sizes and LaTeX pagination).
Without images (placeholder boxes): **22–26 pages**.
