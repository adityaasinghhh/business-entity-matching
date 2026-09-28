from pathlib import Path
import subprocess
import sys

root = Path(r"C:\Users\adity\hackathon")

candidates = [
    root / "utils" / "validate_submission.py",
    root / "student_resource" / "utils" / "validate_submission.py",
]

validator = next((p for p in candidates if p.exists()), None)

if validator is None:
    print("OFFICIAL VALIDATOR NOT FOUND")
    print("Checked:")
    for p in candidates:
        print(" ", p)
    sys.exit(1)

matching = root / "outputs" / "final" / "matching_results.tsv"
candidate = root / "outputs" / "final" / "candidate_pairs.tsv"
test_dir = root / "data" / "raw" / "train" / "train" / "test" / "test"

print("=" * 75)
print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
print("=" * 75)
print("Validator :", validator)
print("Matching  :", matching)
print("Candidate :", candidate)
print("Test dir  :", test_dir)
print()

cmd = [
    sys.executable,
    str(validator),
    "--matching", str(matching),
    "--candidate", str(candidate),
    "--test-dir", str(test_dir),
]

print("COMMAND:")
print(" ".join(f'"{x}"' for x in cmd))
print()

result = subprocess.run(cmd)
sys.exit(result.returncode)
