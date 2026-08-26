"""Retrain all three final models using the leakage-safe Option B procedure."""
import subprocess, sys

modules = [
    "models.member1_logistic_regression",
    "models.member2_svm",
    "models.member3_random_forest",
]
for module in modules:
    print(f"\n=== Training {module} ===")
    subprocess.run([sys.executable, "-m", module], check=True)
print("\nTraining complete. Run compare_models.py and run_behavioral_tests.py next.")
