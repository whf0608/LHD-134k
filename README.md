# Instantaneous Multi-Hazard Disaster Response: Damaged Building Detection from Single Post-Disaster HRRS Imagery for Rapid Global Mapping in Data-Sparse Regions Without Pre-Disaster Baselines



---

### 🌍 **Project Overview**
This project addresses the critical challenge of **rapid damaged building detection** in post-disaster scenarios using single high-resolution remote sensing (HRRS) imagery. We introduce the **world's largest globally distributed disaster-damaged building dataset (LHD-134k)** and a novel model **DBSeg-SAM** designed for zero-dependency on pre-disaster baselines, enabling instant mapping in data-sparse regions.

### 📌 **Key Innovations**
- **LHD-134k Dataset**:
  - **Scale & Diversity**: 134,000+ labeled HRRS images covering 12 disaster types across 106 global regions
  - **Resolution**: 0.1-1m spatial resolution for precise building-level damage assessment
  - **Zero-Baseline Design**: Enables detection using only post-disaster imagery

- **DBSeg-SAM Model**:
  - Integrates damaged building features into the SAM-CLIP framework
  - Achieves state-of-the-art accuracy and robustness across unseen disaster scenarios
  - Trained exclusively on LHD-134k for maximum generalization

### 🔬 **Technical Highlights**


### 📊 **Experimental Results**
| Disaster Type | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Earthquake   | 0.92      | 0.89   | 0.90     |
| Flood        | 0.88      | 0.91   | 0.89     |
| Wildfire      | 0.90      | 0.87   | 0.88     |
| ...          | ...       | ...    | ...      |

**Generalization Performance**:  
DBSeg-SAM achieves **+15% F1-score improvement** over baselines in cross-region validation.

### 🛠️ **Installation & Usage**
```bash
# Clone repository
git clone https://github.com/your-username/repo-name.git
cd repo-name

# Install dependencies
pip install -r requirements.txt

# Run inference
python detect.py --image path/to/post_disaster.tif --model_path models/dbseg_sam.pth
```

### 👥 **Authors**
| Name                | Affiliation                      | Email                 |
|---------------------|----------------------------------|-----------------------|
| Haifeng Wang         | Wuhan University                | wanghaifeng68@whu.edu.cn |
| Wei He (Corr.)       | Wuhan University                | weihe1990@whu.edu.cn   |
| Naoto Yokoya        | University of Tokyo & RIKEN AIP | yokoya@k.u-tokyo.ac.jp|

### 📜 **Citation**
```bibtex
@article{wang2025instantaneous,
  title={Instantaneous Multi-Hazard Disaster Response: Damaged Building Detection from Single Post-Disaster HRRS Imagery},
  author={Wang, Haifeng and He, Wei and Yokoya, Naoto},
  journal={Journal of Remote Sensing},
  year={2025},
  publisher={Springer}
}
```

### 🤝 **Contribution**
Contributions are welcome! Please follow our [contribution guidelines](CONTRIBUTING.md). For major changes, open an issue first to discuss your proposed changes.

---

*© 2025 State Key Laboratory of Information Engineering in Surveying, Wuhan University. Licensed under MIT.*
```
