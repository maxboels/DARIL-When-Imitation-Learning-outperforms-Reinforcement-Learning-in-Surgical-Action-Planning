#!/bin/bash
echo "🔧 Compiling MICCAI paper..."

# Clean previous files
rm -f *.aux *.bbl *.blg *.log *.out

# Compile
pdflatex complete_paper.tex
bibtex complete_paper
pdflatex complete_paper.tex
pdflatex complete_paper.tex

if [ -f "complete_paper.pdf" ]; then
    echo "✅ Paper compiled successfully!"
    echo "📄 File: complete_paper.pdf"
else
    echo "❌ Compilation failed. Check the log."
fi

# Clean up
rm -f *.aux *.bbl *.blg *.log *.out
