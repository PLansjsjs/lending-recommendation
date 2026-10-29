"""Run the whole pipeline in order.

    python run_all.py            -> uses the raw Lending Club file in data/
    python run_all.py --demo     -> first creates SYNTHETIC test data (for a trial run only)
    python run_all.py data\\loan.csv   -> name the raw file explicitly
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["00_make_sample.py", "01_clean.py", "02_eda.py", "03_regression.py",
         "04_classification.py", "05_clustering.py", "06_recommend.py", "demo_borrower.py"]


def run(script, args=()):
    print("\n" + "#" * 70 + f"\n# {script}\n" + "#" * 70, flush=True)
    start = time.time()
    code = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT).returncode
    if code != 0:
        print(f"\n{script} failed (exit code {code}). Fix the error above, then run it again.")
        sys.exit(code)
    print(f"[{script} finished in {time.time() - start:.0f}s]")


def main():
    args = [a for a in sys.argv[1:] if a != "--demo"]
    if "--demo" in sys.argv:
        run("make_demo_data.py")
    for step in STEPS:
        run(step, args if step == "00_make_sample.py" else ())
    print("\nAll done. Look in the outputs/ folder (figures, tables, models).")


if __name__ == "__main__":
    main()
