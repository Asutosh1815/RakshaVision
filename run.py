"""
RakshaVision Launcher Script.
Run directly using:
    python run.py
or
    python app.py
or
    streamlit run app.py
"""

import os
import sys
import subprocess

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.py"))
    cmd = [sys.executable, "-m", "streamlit", "run", app_path] + sys.argv[1:]
    print("=" * 65)
    print("[+] Launching RakshaVision AI Industrial Safety Command Center...")
    print(f"[*] Executing: {' '.join(cmd)}")
    print("=" * 65)
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        print("\nRakshaVision stopped by user.")

if __name__ == "__main__":
    main()
