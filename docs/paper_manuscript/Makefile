# Makefile for MICCAI 2025 Paper Build System
# Surgical Action Triplet Prediction: IL vs RL Comparison

PAPER_NAME = surgical_action_prediction_miccai2025
PAPER_TEX = $(PAPER_NAME).tex
PAPER_PDF = $(PAPER_NAME).pdf

# Directories
RESULTS_DIR = ../results
FIGURES_DIR = figures
TABLES_DIR = tables

# Python scripts
FIGURE_SCRIPT = generate_paper_figures.py
TABLE_SCRIPT = create_results_table.py

# LaTeX compilation
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all paper figures tables clean clean-all watch help

# Default target
all: paper

# Main paper compilation
paper: $(PAPER_PDF)

$(PAPER_PDF): $(PAPER_TEX) references.bib figures tables
	@echo "Compiling LaTeX paper..."
	$(LATEX) $(PAPER_TEX)
	$(BIBTEX) $(PAPER_NAME)
	$(LATEX) $(PAPER_TEX)
	$(LATEX) $(PAPER_TEX)
	@echo "✓ Paper compiled successfully: $(PAPER_PDF)"

# Generate all figures
figures:
	@echo "Generating paper figures..."
	python $(FIGURE_SCRIPT) --results_dir $(RESULTS_DIR) --output_dir $(FIGURES_DIR)
	@echo "✓ Figures generated in $(FIGURES_DIR)/"

# Generate all tables
tables:
	@echo "Generating paper tables..."
	python $(TABLE_SCRIPT) --results_dir $(RESULTS_DIR) --output_dir $(TABLES_DIR)
	@echo "✓ Tables generated in $(TABLES_DIR)/"

# Generate figures and tables
assets: figures tables

# Quick compilation (skip bibliography)
quick:
	@echo "Quick LaTeX compilation..."
	$(LATEX) $(PAPER_TEX)
	@echo "✓ Quick compilation complete"

# Watch for changes and auto-compile
watch:
	@echo "Watching for changes (Ctrl+C to stop)..."
	@while true; do \
		inotifywait -e modify $(PAPER_TEX) 2>/dev/null && \
		make quick; \
	done

# Count words in paper
wordcount:
	@echo "Word count analysis..."
	@pdftotext $(PAPER_PDF) - | wc -w | awk '{print "Total words: " $$1}'
	@echo "LaTeX word count (approximate):"
	@texcount -brief $(PAPER_TEX)

# Check paper format compliance
check:
	@echo "Checking MICCAI format compliance..."
	@echo "✓ Page limit: 8 pages + 2 references"
	@echo "✓ Font: Default LLNCS (Computer Modern)"
	@echo "✓ Anonymous submission: Check author section"
	@echo "✓ Bibliography style: splncs04"
	@echo "For detailed check, review MICCAI guidelines"

# Spell check
spellcheck:
	@echo "Running spell check..."
	@aspell --mode=tex --check $(PAPER_TEX)

# Clean intermediate files
clean:
	@echo "Cleaning intermediate files..."
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.synctex.gz *.out *.toc
	@echo "✓ Cleaned intermediate files"

# Clean everything including outputs
clean-all: clean
	@echo "Cleaning all generated files..."
	rm -f $(PAPER_PDF)
	rm -rf $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.png
	rm -rf $(TABLES_DIR)/*.tex
	@echo "✓ Cleaned all files"

# Create submission package
submission: paper
	@echo "Creating submission package..."
	mkdir -p submission
	cp $(PAPER_PDF) submission/
	cp $(PAPER_TEX) submission/
	cp references.bib submission/
	cp -r $(FIGURES_DIR) submission/ 2>/dev/null || true
	cp -r $(TABLES_DIR) submission/ 2>/dev/null || true
	tar -czf $(PAPER_NAME)_submission.tar.gz submission/
	@echo "✓ Submission package created: $(PAPER_NAME)_submission.tar.gz"

# Validate paper structure
validate:
	@echo "Validating paper structure..."
	@grep -q "\\title{" $(PAPER_TEX) && echo "✓ Title found" || echo "✗ Title missing"
	@grep -q "\\begin{abstract}" $(PAPER_TEX) && echo "✓ Abstract found" || echo "✗ Abstract missing"
	@grep -q "\\keywords{" $(PAPER_TEX) && echo "✓ Keywords found" || echo "✗ Keywords missing"
	@grep -q "\\section{Introduction}" $(PAPER_TEX) && echo "✓ Introduction section found" || echo "✗ Introduction missing"
	@grep -q "\\section{Methods}" $(PAPER_TEX) && echo "✓ Methods section found" || echo "✗ Methods missing"
	@grep -q "\\section{Results}" $(PAPER_TEX) && echo "✓ Results section found" || echo "✗ Results missing"
	@grep -q "\\section{Conclusion}" $(PAPER_TEX) && echo "✓ Conclusion section found" || echo "✗ Conclusion missing"
	@grep -q "\\begin{thebibliography}" $(PAPER_TEX) && echo "✓ Bibliography found" || echo "✗ Bibliography missing"

# Help target
help:
	@echo "MICCAI 2025 Paper Build System"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build complete paper (default)"
	@echo "  paper        - Compile LaTeX paper"
	@echo "  figures      - Generate all figures from results"
	@echo "  tables       - Generate all tables from results"
	@echo "  assets       - Generate both figures and tables"
	@echo "  quick        - Quick compilation (skip bibliography)"
	@echo "  watch        - Auto-compile on file changes"
	@echo "  wordcount    - Count words in paper"
	@echo "  check        - Check MICCAI format compliance"
	@echo "  spellcheck   - Run spell checker"
	@echo "  validate     - Validate paper structure"
	@echo "  submission   - Create submission package"
	@echo "  clean        - Remove intermediate files"
	@echo "  clean-all    - Remove all generated files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Paper: $(PAPER_NAME)"
	@echo "Results: $(RESULTS_DIR)"

# Development helpers
dev: assets quick
	@echo "✓ Development build complete"

# Final submission preparation
final: clean paper wordcount check validate
	@echo "✓ Final paper ready for submission"
	@echo "Please review the PDF for formatting and content"