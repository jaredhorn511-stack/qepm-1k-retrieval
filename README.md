# QEPM-1K: Sub-Linear Knowledge Retrieval Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17848904.svg)](https://doi.org/10.5281/zenodo.17848904)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Published Research

**Paper:** Horn, J.P. (2025). Sub-Linear Knowledge Retrieval via Quantum-Inspired 
Hyperdimensional Folded Space. Zenodo. https://doi.org/10.5281/zenodo.17848904

---

## 📦 What's Included

This repository contains the complete implementation of the QEPM-1K knowledge retrieval system described in the paper.

**Core Files:**

1. **build_qepm_1k.py** - Builds 1,100 Q&A knowledge base
2. **test_1k_folded_space.py** - Tests with 4D folded space indexing
3. **quantum_hdc_encoder_optimized.py** - 10,000D HDC encoder
4. **quantum_inference_optimized_v2.py** - Pattern inference engine
5. **entangle_storage_fixed.py** - PHOTON storage format
6. **quantum_training_ultra.py** - Training system (203 patterns/sec)
7. **hdc_utils.py** - Core HDC utilities
8. **deployment_guide.py** - Technical documentation

**Documentation:**

9. **README.md** - This file
10. **EXECUTIVE_SUMMARY.md** - Performance overview

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install numpy numba
```

### Build Knowledge Base

```bash
# Build 1,100 Q&A knowledge base (~24 seconds)
python build_qepm_1k.py
```

**Output:** `scaling/qepm_knowledge_1k_output/qepm_knowledge_1k/`

### Test Retrieval

```bash
# Test with 4D folded space indexing
python test_1k_folded_space.py
```

**Expected Results:**
- Accuracy: 100% (15/15 queries)
- Average speed: 0.88ms
- Exact bucket hits: 93%

---

## 📊 Reproducing Paper Results

### Main Result (Table 1)

```bash
python test_1k_folded_space.py
```

**Expected output:**
```
Overall Performance:
- Accuracy: 100.0% (15/15)
- Average time: 0.88ms
- Median time: 0.78ms
- Throughput: 1,140 queries/second

Folded Space Strategy:
- Exact bucket: 93% (14/15)
- 1-hop neighbors: 7% (1/15)
- Full search: 0% (0/15)
```

### Hardware Requirements

**Minimum:**
- CPU: Intel Celeron N4020 @ 1.1GHz (or equivalent)
- RAM: 8 GB
- Storage: 1 GB available
- OS: Windows/Linux/macOS

**Note:** No GPU required!

---

## 📁 File Structure

```
QEPM-1K-Code/
├── README.md                              # This file
├── EXECUTIVE_SUMMARY.md                   # Performance overview
│
├── build_qepm_1k.py                       # Build knowledge base
├── test_1k_folded_space.py               # Test & validate
│
├── quantum_hdc_encoder_optimized.py      # HDC encoder (10,000D)
├── quantum_inference_optimized_v2.py     # Inference engine
├── entangle_storage_fixed.py             # Storage system
├── quantum_training_ultra.py             # Training (203 pat/sec)
├── hdc_utils.py                          # Core utilities
│
└── deployment_guide.py                    # Technical docs
```

---

## 🔧 Usage Examples

### Example 1: Build Custom Knowledge Base

```python
from build_qepm_1k import build_knowledge_base

# Create your own Q&A pairs
data = [
    ("What is AI?", "Artificial Intelligence is..."),
    ("What is ML?", "Machine Learning is..."),
    # ... add more
]

# Build knowledge base
build_knowledge_base(
    data=data,
    output_path="my_knowledge_base/",
    model_dim=2048
)
```

### Example 2: Query Knowledge Base

```python
from test_1k_folded_space import FoldedSpaceKnowledgeQEPM

# Load knowledge base
kb = FoldedSpaceKnowledgeQEPM(
    model_path="scaling/qepm_knowledge_1k_output/qepm_knowledge_1k/"
)

# Query
answer, confidence, strategy = kb.query("What is machine learning?")

print(f"Answer: {answer}")
print(f"Confidence: {confidence:.2%}")
print(f"Strategy: {strategy}")
```

---

## 📈 Performance Benchmarks

**On Intel Celeron N4020 @ 1.1GHz:**

| Metric | Value |
|--------|-------|
| Knowledge pairs | 1,100 |
| Accuracy | 100% (15/15 test queries) |
| Average speed | 0.88ms |
| Median speed | 0.78ms |
| Min speed | 0.59ms |
| Max speed | 1.30ms |
| Throughput | 1,140 queries/second |
| Exact bucket hits | 93% (O(1) retrieval) |

**Comparison:**
- vs. Exhaustive search: **162× faster**
- vs. 80-pair baseline: **13× faster** with **13.75× more data**

---

## 🧪 Testing

Run all tests:

```bash
# Test HDC encoder
python quantum_hdc_encoder_optimized.py

# Test inference engine
python quantum_inference_optimized_v2.py

# Test storage
python entangle_storage_fixed.py

# Test complete system
python test_1k_folded_space.py
```

---

## 🛠️ Customization

### Change Knowledge Base Size

Edit `build_qepm_1k.py`:

```python
# Change from 1,100 to your desired size
PAIRS_PER_DOMAIN = {
    'ml_ai': 100,           # Change these numbers
    'computer_science': 100,
    # ...
}
```

### Change Folded Space Resolution

Edit `test_1k_folded_space.py`:

```python
# Change from 7×7×7×7 to different size
self.grid_size = 7  # Try 5, 9, 11, etc.
```

**Recommendation:** 
- Small KB (<500): grid_size = 5
- Medium KB (500-2000): grid_size = 7
- Large KB (2000+): grid_size = 9

---

## 📖 Citation

If you use this code, please cite:
```bibtex
@misc{horn2025sublinear,
  title={Sub-Linear Knowledge Retrieval via Quantum-Inspired Hyperdimensional Folded Space},
  author={Horn, Jared Paul},
  year={2025},
  month={December},
  publisher={Zenodo},
  doi={10.5281/zenodo.17848904},
  url={https://doi.org/10.5281/zenodo.17848904}
}
```

---

## 📧 Contact

**Jared Paul Horn**  
Email: jaredhorn511@gmail.com  
DOI: [10.5281/zenodo.17848904](https://doi.org/10.5281/zenodo.17848904)

---

## 📄 License

This code is provided for research and educational purposes.

**You are free to:**
- Use the code for research
- Modify and adapt
- Include in your projects

**Please:**
- Cite the paper if you use this code
- Share improvements back to the community

---

## 🙏 Acknowledgments

This work was completed on consumer hardware (Intel Celeron N4020, 12GB RAM) to demonstrate that advanced AI research doesn't require expensive GPUs.

---

## 🐛 Known Issues

None currently. If you find bugs, please contact: jaredhorn511@gmail.com

---

## 📝 Changelog

**v1.0 (December 2025)**
- Initial release
- 1,100 Q&A knowledge base
- 0.88ms average retrieval
- 100% accuracy on test set

---

## ⚡ Quick Reference

**Build knowledge base:**
```bash
python build_qepm_1k.py
```

**Test system:**
```bash
python test_1k_folded_space.py
```

**Expected results:**
- 100% accuracy
- 0.88ms average speed
- 93% exact bucket hits

**Hardware:**
- Consumer laptop (no GPU)
- ~$200 cost
- 12GB RAM sufficient

**Performance:**
- 162× faster than exhaustive search
- 13× faster than 80-pair baseline
- 1,140 queries/second throughput

---

**For questions or issues, contact: jaredhorn511@gmail.com**

---

## ⭐ Found This Useful?

If this research helped you or you'd like to support open science, please star this repository!

Questions? Open an issue or reach out: jaredhorn511@gmail.com
