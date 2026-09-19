"""
Training & Evaluation Script for Mendeley PPE Dataset in RakshaVision.
Trains or fine-tunes YOLOv8 on the 2,286-image Mendeley PPE dataset.
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO on Mendeley PPE Dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    args = parser.parse_args()

    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset", "mendeley_ppe2286", "data.yaml"))
    if not os.path.exists(yaml_path):
        print(f"[Error] data.yaml not found at: {yaml_path}")
        sys.exit(1)

    print("=" * 65)
    print("🛡️  RakshaVision - Training YOLO on Mendeley PPE Dataset")
    print(f"[*] Dataset: {yaml_path}")
    print(f"[*] Base Model: {args.base_model}")
    print(f"[*] Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
    print("=" * 65)

    model = YOLO(args.base_model)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/mendeley_train",
        name="ppe_model",
        exist_ok=True
    )

    print("[*] Training completed. Best weights saved in runs/mendeley_train/ppe_model/weights/best.pt")

if __name__ == "__main__":
    main()
