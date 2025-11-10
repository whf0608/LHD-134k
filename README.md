# LHD-134k: <small>A Global Multi-Hazard Dataset for Post-Disaster Building Damage Detection in High-Resolution Remote Sensing Imagery</small>  

---
### 🌍 **Home** ([🏠](http://frp9.aaszxc.asia:12365/lhd134k/))

### 🌍 **Project Overview**
This project addresses the critical challenge of **rapid damaged building detection** in post-disaster scenarios using single high-resolution remote sensing (HRRS) imagery. We introduce the **world's largest globally distributed disaster-damaged building dataset (LHD-134k)** and a novel model **DBSeg-SAM** designed for zero-dependency on pre-disaster baselines, enabling instant mapping in data-sparse regions. 
### 📌 **Key Innovations**
- **LHD-134k Dataset**:
  - **Scale & Diversity**: 134,000+ labeled HRRS images covering 12 disaster types across 106 global regions
  - **Resolution**: 0.1-1m spatial resolution for precise building-level damage assessment
  - **Baseline Design**: Enables detection using only post-disaster imagery

- **DBSeg-SAM Model**:
  - Integrates damaged building features into the SAM-CLIP framework
  - Achieves state-of-the-art accuracy and robustness across unseen disaster scenarios
  - Trained exclusively on LHD-134k for maximum generalization

### 🔬 **LHD-134k Dataset**  ([Download📥](https://github.com/whf0608/LHD-134k/releases/download/lhd134k/disasters_db_dataset_part0001.zip))

<img width="100%" height="100%" alt="image" src="https://github.com/user-attachments/assets/141a7a66-aee8-4eff-8ea4-1e459caf99e3" />

#### 1. Down Data
```bash
# data url
cd disaster_dataset_data_down_url_txt\Earthquake\Afghanistan_Earthquake\time
# downing
wget -i post.txt
```

#### 2. Data Location
<img width="100%" height="100%" alt="image" src="https://github.com/user-attachments/assets/b8a40b02-4b9b-4b51-917c-368202c63675" />

[View](https://github.com/whf0608/LHD-134k/blob/main/disaster_dataset_data_down_url_txt/point_geojson.geojson)

#### 3. Make Tile
```bash
gdal2tiles.bat -p mercator  -z 5-20 -w leaflet -r average -a 0.0 /path/to/disaster_post.tid /path/save/
```
#### 4. Labeling Data
[Qgis tools](https://qgis.org/download/)

#### 5. Make Dataset
```bash
cd disaster_dataset_data_down_url_txt
# runing make dataset
python make_datatset.py
```


### 🚀 **Model**

<img width="100%" height="100%" alt="image"  alt="image" src="https://github.com/user-attachments/assets/a968dc64-ce2f-444f-a7ba-faf392450a71" />



**Generalization Performance**:  
DBSeg-SAM achieves **+15% F1-score improvement** over baselines in cross-region validation.

### 🛠️ **Installation & Usage**
```bash
# Clone repository
git clone https://github.com/whf0608/LHD-134k.git
cd LHD-134k

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




---

*© 2025 State Key Laboratory of Information Engineering in Surveying, Wuhan University.
```
