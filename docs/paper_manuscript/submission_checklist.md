# MICCAI 2025 Submission Checklist
## Surgical Action Triplet Prediction: IL vs RL Comparison

### Pre-Submission Requirements

#### 📄 Paper Format Compliance
- [ ] **Page limit**: Maximum 8 pages main content + 2 pages references
- [ ] **Template**: Using official MICCAI 2025 LLNCS template
- [ ] **Font**: Default Computer Modern (no changes)
- [ ] **Margins**: Default LLNCS margins (no modifications)
- [ ] **Font size**: Default 10pt (tables minimum 8pt)
- [ ] **Anonymization**: Author section anonymized properly
- [ ] **No inline figures**: Text wrapping around figures prohibited

#### 📝 Content Requirements
- [ ] **Title**: Clear and descriptive
- [ ] **Abstract**: 150-250 words
- [ ] **Keywords**: At least one keyword provided
- [ ] **Introduction**: Research motivation and contributions
- [ ] **Methods**: Clear methodology description
- [ ] **Results**: Quantitative results with tables/figures
- [ ] **Discussion**: Analysis of findings
- [ ] **Conclusion**: Summary and future work
- [ ] **References**: Properly formatted with splncs04 style

#### 🔍 Anonymization Check
- [ ] **Author names**: Removed or anonymized
- [ ] **Affiliations**: Anonymized
- [ ] **Self-citations**: Referred to in third person
- [ ] **Dataset location**: Masked or generalized
- [ ] **Code repositories**: Anonymized if mentioned
- [ ] **Acknowledgments**: Removed for initial submission
- [ ] **Funding information**: Removed for initial submission

#### 📊 Results and Evaluation
- [ ] **Performance metrics**: mAP scores for all methods
- [ ] **Statistical significance**: P-values and effect sizes
- [ ] **Baseline comparisons**: Comparison with existing methods
- [ ] **Multi-horizon evaluation**: Planning performance analysis
- [ ] **Scenario-specific analysis**: Complex vs routine surgery
- [ ] **Computational efficiency**: Runtime and memory analysis
- [ ] **Clinical relevance**: Discussion of practical implications

#### 🖼️ Figures and Tables
- [ ] **Figure quality**: High resolution (300 DPI minimum)
- [ ] **Figure format**: PDF or EPS preferred
- [ ] **Figure captions**: Descriptive and placed below figures
- [ ] **Table captions**: Descriptive and placed above tables
- [ ] **Label visibility**: All text readable at paper scale
- [ ] **Color scheme**: Colorblind-friendly palette
- [ ] **Figure numbering**: Sequential and referenced in text

### Technical Validation

#### 🧪 Experimental Rigor
- [ ] **Dataset**: CholecT50 with proper train/test split
- [ ] **Reproducibility**: Clear experimental setup
- [ ] **Fair comparison**: Same evaluation protocol for all methods
- [ ] **Multiple runs**: Statistical validity of results
- [ ] **Hyperparameter reporting**: All important parameters listed
- [ ] **Ablation studies**: Key design choices validated

#### 🔬 Method Contributions
- [ ] **Novel IL baseline**: Autoregressive approach description
- [ ] **RL variants**: Multiple RL approaches compared
- [ ] **IRL enhancement**: Inverse RL methodology
- [ ] **Evaluation framework**: Recognition vs planning metrics
- [ ] **Clinical insights**: When to use IL vs RL

#### 📈 Results Analysis
- [ ] **Main findings**: IL excels at recognition, RL helps in complex scenarios
- [ ] **Performance numbers**: 0.979 mAP for IL baseline reported
- [ ] **Planning degradation**: Multi-step prediction analysis
- [ ] **Computational trade-offs**: Efficiency vs accuracy discussion
- [ ] **Failure cases**: Limitations and failure modes discussed

### Submission Package

#### 📦 Required Files
- [ ] **Main paper PDF**: surgical_action_prediction_miccai2025.pdf
- [ ] **Source files**: LaTeX source if requested
- [ ] **Figures**: All figures in high resolution
- [ ] **Supplementary material**: If any (multimedia only)

#### 🔗 Optional Materials
- [ ] **Code availability**: Anonymized repository if mentioned
- [ ] **Dataset access**: Instructions for data access
- [ ] **Reproducibility package**: Scripts and configurations

### Pre-Submission Testing

#### ⚡ Build System
- [ ] **Clean build**: `make clean && make all` works
- [ ] **Figure generation**: `make figures` generates all plots
- [ ] **Table generation**: `make tables` creates LaTeX tables
- [ ] **Word count**: Within MICCAI limits
- [ ] **Spell check**: No spelling errors
- [ ] **Reference formatting**: All citations properly formatted

#### 🔍 Final Review
- [ ] **Grammar check**: Professional writing quality
- [ ] **Technical accuracy**: All claims supported by results
- [ ] **Figure references**: All figures referenced in text
- [ ] **Table references**: All tables referenced in text
- [ ] **Citation completeness**: All claims properly cited
- [ ] **Consistency**: Terminology used consistently

### Submission Timeline

#### 📅 Week Before Submission
- [ ] **Content complete**: All sections written
- [ ] **Results finalized**: All experiments completed
- [ ] **Figures generated**: All plots created and polished
- [ ] **Tables formatted**: All results tables ready
- [ ] **Internal review**: Co-authors review complete

#### 📅 3 Days Before Deadline
- [ ] **Final draft**: Paper content frozen
- [ ] **Formatting check**: All MICCAI requirements met
- [ ] **Anonymization verification**: Double-check anonymization
- [ ] **Submission package**: All files ready

#### 📅 Day of Submission
- [ ] **Final PDF**: Final version generated
- [ ] **File size check**: PDF under size limits
- [ ] **Upload test**: Test submission system
- [ ] **Backup copies**: Multiple copies saved
- [ ] **Submission confirmation**: Receipt confirmed

### Post-Submission

#### 📧 After Submission
- [ ] **Confirmation email**: Submission receipt received
- [ ] **Submission ID**: Record submission number
- [ ] **Backup storage**: Archive submission files
- [ ] **Review preparation**: Prepare for reviewer comments

### Quality Metrics

#### 📊 Target Metrics
- [ ] **Recognition mAP**: >0.95 for IL baseline
- [ ] **Planning degradation**: <20% at 5-step horizon
- [ ] **RL improvement**: Demonstrable in complex scenarios
- [ ] **Computational efficiency**: Real-time inference capability
- [ ] **Clinical relevance**: Clear application pathways

#### 🎯 Success Criteria
- [ ] **Technical contribution**: Novel IL vs RL comparison
- [ ] **Clinical impact**: Practical surgical AI insights
- [ ] **Methodological rigor**: Sound experimental design
- [ ] **Reproducibility**: Clear implementation details
- [ ] **Broader impact**: Implications for surgical AI field

---

## Quick Commands

```bash
# Generate all assets
make assets

# Build paper
make paper

# Final check
make final

# Create submission package
make submission
```

## Contact Information

- **Conference**: MICCAI 2025
- **Submission deadline**: [Check MICCAI website]
- **Page limit**: 8 + 2 references
- **Template**: LLNCS format

## Notes

- Review MICCAI guidelines regularly for updates
- Keep submission anonymous until acceptance
- Prepare for potential reviewer requests
- Consider clinical reviewers' perspectives