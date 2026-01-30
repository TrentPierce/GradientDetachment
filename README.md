# GradientDetachment: Neural ODE Cryptanalysis Framework

## ⚠️ ACADEMIC RESEARCH PROJECT - DUAL USE NOTICE

This repository contains academic research on the resistance of ARX ciphers to Neural ODE-based cryptanalysis. **This is a NEGATIVE RESULT** - we demonstrate that Neural ODEs FAIL to break ARX ciphers due to mathematical barriers (Gradient Detachment).

### Safety Assessment: LOW RISK
- Demonstrates that ML methods **fail** on modern ARX designs
- All cipher operations use smooth approximations (not exact cryptanalysis)
- Does not provide tools for breaking real-world encryption
- Educational value: Validates ARX design choices

**Full assessment:** See [DUAL_USE_SAFETY.md](DUAL_USE_SAFETY.md)

---

## 🎯 Research Question

**Can Neural ODEs break ARX ciphers?** 

**Answer:** No. Our research demonstrates that ARX ciphers (like Speck) are fundamentally resistant to Neural ODE-based attacks due to the "Gradient Detachment" phenomenon caused by modular arithmetic operations.

---

## 🔬 Key Findings

### 1. Gradient Detachment Phenomenon
ARX cipher operations create a "sawtooth" loss landscape that prevents gradient-based learning:
- **Modular addition** creates discontinuities
- **Rotation operations** create periodic gradient changes  
- **Combined effect**: Gradient flow breaks down

### 2. Cross-Cipher Comparison Results

| Cipher Family | 1-Round Accuracy | 4-Round Accuracy | Security |
|--------------|------------------|------------------|----------|
| **ARX (Speck)** | ~2.5% | 0% | ✅ Strongest |
| **Feistel** | ~15% | 0% | ⚠️ Weaker at low rounds |
| **SPN** | ~12% | 0% | ⚠️ Intermediate |

### 3. Round Security Threshold
All cipher families achieve 0% accuracy at **4+ rounds**, demonstrating the security of modern designs.

---

## 🚀 Quick Start

### One-Command Verification
```bash
python reproduce_sawtooth.py
```

This runs both tests:
1. **Single-batch test**: 100% accuracy (proves no implementation bugs)
2. **Multi-sample test**: ~2.5% accuracy (demonstrates Gradient Detachment)

### Installation
```bash
pip install -e .
```

### Run Experiments
```bash
# Test cross-cipher comparison
python experiments/test_cross_cipher_comparison.py

# Test round security
python experiments/test_round_security.py

# Differential distinguisher
python experiments/test_distinguisher.py
```

---

## 📁 Project Structure

```
GradientDetachment/
├── README.md                          # This file
├── RESEARCH_PAPER.md                  # Full academic paper
├── DUAL_USE_SAFETY.md                 # Safety assessment
├── LICENSE                            # MIT License
├── reproduce_sawtooth.py              # One-command verification
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
├── src/ctdma/                         # Core framework
│   ├── ciphers/                       # Cipher implementations
│   │   ├── speck.py                  # ARX cipher
│   │   ├── feistel.py                # Feistel cipher
│   │   └── spn.py                    # SPN cipher
│   ├── neural_ode/                    # Neural ODE solver
│   └── attacks/                       # Cryptanalysis methods
├── experiments/                       # Test scripts
├── debug_scripts/                     # Verification scripts
└── docs/                              # Documentation
```

---

## 🔍 Why This Matters

### For Cryptographers
- **Validates ARX design**: Confirms ARX ciphers are ML-resistant
- **Guidance**: Use ARX + 4+ rounds for ML-resistant designs
- **Understanding**: Explains why differentiable cryptanalysis fails on ARX

### For ML Researchers
- **Negative result**: Saves time exploring Neural ODEs for ARX
- **Mathematical insight**: Reveals limitations of gradient-based methods
- **Cross-domain**: Demonstrates cipher structure impacts learnability

### For the Community
- **Transparency**: Open research on cryptographic ML resistance
- **Education**: Helps understand cipher design principles
- **Ethics**: Demonstrates responsible disclosure of cryptanalysis research

---

## 📊 Reproducibility

All results are reproducible:
- Fixed random seeds where appropriate
- Documented hyperparameters
- One-command verification script
- All experimental code included

Run `python reproduce_sawtooth.py` to verify our findings.

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{gradientdetachment2026,
  title={Gradient Detachment in Continuous-Time Cryptanalysis: 
         The Topological Resistance of ARX Ciphers},
  author={[Authors]},
  year={2026},
  note={Demonstrates Neural ODEs fail on ARX due to modular arithmetic barriers}
}
```

---

## 🤝 Contributing

This is an academic research project. For issues or questions, please open a GitHub issue.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## ⚖️ Ethical Use

This research demonstrates **negative results** (attacks fail). It is intended for:
- Academic research
- Educational purposes
- Cipher design validation
- Understanding ML limitations

**Not intended for:** Breaking real-world encryption systems.

See [DUAL_USE_SAFETY.md](DUAL_USE_SAFETY.md) for full safety assessment.

---

## 📧 Contact

For academic inquiries: [Contact Information]

---

**Research Status:** ✅ Complete - Ready for Publication

**Venue Targets:** CRYPTO 2026, IEEE S&P 2026, or similar top-tier venues

**Key Contribution:** First systematic demonstration that ARX ciphers resist Neural ODE attacks through Gradient Detachment