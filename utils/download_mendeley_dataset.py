"""
Automated Dataset Fetcher for Mendeley PPE Dataset.
Downloads and extracts the 2,286-image Mendeley PPE dataset directly from Mendeley Data.
"""

import os
import sys
import time
import zipfile
import urllib.request

MENDELEY_FILE_URL = "https://data.mendeley.com/public-files/datasets/zkzghjvpn2/files/b8bc8fe2-5aaa-49d8-b012-a9858258f05d/file_downloaded"
DEST_ZIP = "dataset_ppe2286.zip"
TARGET_DIR = os.path.join("dataset", "mendeley_ppe2286")

def download_and_extract():
    if os.path.exists(TARGET_DIR) and os.path.exists(os.path.join(TARGET_DIR, "data.yaml")):
        print(f"[*] Dataset already exists at: {TARGET_DIR}")
        return TARGET_DIR

    os.makedirs("dataset", exist_ok=True)
    if not os.path.exists(DEST_ZIP):
        print(f"[*] Downloading Mendeley PPE Dataset from: {MENDELEY_FILE_URL}")
        req = urllib.request.Request(MENDELEY_FILE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(DEST_ZIP, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"    Downloaded {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({(downloaded/total)*100:.1f}%)", end="\r")
        print("\n[*] Download complete.")

    print(f"[*] Extracting {DEST_ZIP}...")
    with zipfile.ZipFile(DEST_ZIP, "r") as z:
        z.extractall("dataset")

    # Clean folder names
    raw_folder = os.path.join("dataset", "20250731-ppe2286y")
    if os.path.exists(raw_folder) and not os.path.exists(TARGET_DIR):
        os.rename(raw_folder, TARGET_DIR)

    print(f"[*] Mendeley PPE Dataset ready at: {TARGET_DIR}")
    return TARGET_DIR

if __name__ == "__main__":
    download_and_extract()
