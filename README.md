# Instantaneous Multi-Hazard Disaster Response
## Damaged Building Detection from Single Post-Disaster HRRS Imagery for Rapid Global Mapping in Data-Sparse Regions Without Pre-Disaster Baselines

[![GitHub Stars](https://img.shields.io/github/stars/whf0608/LHD-134k?style=social)](https://github.com/whf0608/LHD-134k/)


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


# Clone repository
git clone https://github.com/your-username/repo-name.git
cd repo-name

# Install dependencies
pip install -r requirements.txt

# Run inference
python detect.py --image path/to/post_disaster.tif --model_path models/dbseg_sam.pth
