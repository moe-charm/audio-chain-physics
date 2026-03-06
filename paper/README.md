# Paper Drafts

This folder keeps the public-facing paper draft materials for the project.

Current status:

- research-in-progress
- working manuscript scaffold
- claims intentionally kept conservative

The current framing is:

- this project is an integrated simulator and research tool
- it can already demonstrate measurable degradation under clearly poor cable
  conditions
- it does not yet claim to fully explain every listening impression
- subtle qualities such as veil, treble sheen, extension, soundstage shift, or
  density still require further study

## Files

- `abstract.md`
  Current abstract draft in English.
- `manuscript.md`
  Main manuscript scaffold and draft text.
- `en/main.tex`
  English TeX manuscript draft.
- `ja/main.tex`
  Japanese TeX manuscript draft.

## Writing Position

The recommended claim style for the first paper is:

- not "we proved every cable claim"
- but "we built an integrated physical/circuit/DSP/perceptual simulation
  framework, showed that stressed cable conditions can measurably degrade
  system behavior, and clarified what still remains unexplained"

## Next Steps

- add figures exported from the app
- add measured comparison data
- add ablation tables
- choose a target venue and format

## Build Notes

Suggested starting point:

- English: `paper/en/main.tex`
- Japanese: `paper/ja/main.tex`

Recommended engines:

- English: `pdflatex` or `lualatex`
- Japanese: `lualatex`

Example commands:

```bash
cd paper/en
pdflatex main.tex
pdflatex main.tex
```

```bash
cd paper/ja
lualatex main.tex
lualatex main.tex
```

The current workspace used to prepare these files did not have a LaTeX
distribution installed, so the TeX sources were written but not compiled here.
