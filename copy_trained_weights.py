"""
Post-training script: copies the best.pt from the training run into weights/ 
and shows validation metrics summary.
Run this after train_mendeley_ppe.py completes.
"""
import os
import shutil

SOURCE = "runs/mendeley_train/ppe_model/weights/best.pt"
DEST   = "weights/yolov8n-mendeley-ppe.pt"

if not os.path.exists(SOURCE):
    print(f"[Error] best.pt not found at: {SOURCE}")
    print("  → Training may not have completed yet. Check runs/ directory.")
    exit(1)

shutil.copy2(SOURCE, DEST)
print(f"[✓] Copied {SOURCE} → {DEST}")

# Print metrics if results.csv exists
csv_path = "runs/mendeley_train/ppe_model/results.csv"
if os.path.exists(csv_path):
    import csv
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        # Strip whitespace from keys
        last = {k.strip(): v.strip() for k, v in last.items()}
        print("\n[Training Summary - Last Epoch]")
        for key in ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
            if key in last:
                print(f"  {key}: {last[key]}")
        print(f"\n[✓] New weights loaded at: {DEST}")
        print("[!] Restart the RakshaVision app to use the improved model.")
