# 🦺 Mendeley PPE Dataset (Personal Protective Equipment)

## Dataset Overview
- **Source**: [Mendeley Data Repository (zkzghjvpn2, Version 6)](https://data.mendeley.com/datasets/zkzghjvpn2/6/files/b8bc8fe2-5aaa-49d8-b012-a9858258f05d)
- **Contributors**: Mei-Ling Huang, Ying Cheng
- **Published**: July 28, 2025
- **DOI**: [10.17632/zkzghjvpn2.6](https://doi.org/10.17632/zkzghjvpn2.6)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset Specifications
- **Format**: YOLOv8 / YOLOv11 standard annotation format (`.txt` per `.jpg` image)
- **Image Resolution**: 640 × 640 pixels
- **Total Images**: 2,286
  - **Training Set (`train/`)**: 1,829 images (80%)
  - **Validation Set (`valid/`)**: 457 images (20%)
- **Number of Classes (`nc: 4`)**:
  - `0`: `Helmet` — Worker wearing a protective safety helmet / hard hat
  - `1`: `NoHelmet` — Worker not wearing a helmet
  - `2`: `NoVest` — Worker not wearing a high-visibility safety vest
  - `3`: `Vest` — Worker wearing a high-visibility reflective safety vest

## Usage with YOLOv8 / YOLO11
Train directly with Ultralytics:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="dataset/mendeley_ppe2286/data.yaml", epochs=30, imgsz=640)
```
