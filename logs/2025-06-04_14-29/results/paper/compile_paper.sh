#!/bin/bash
# Compile research paper

echo "Compiling research paper..."

# Compile main paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

# Compile supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex

echo "Paper compilation complete!"
echo "Main paper: paper.pdf"
echo "Supplementary: supplementary.pdf"
