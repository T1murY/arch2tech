# test_sam2_import.py
try:
    from sam2.build_sam import build_sam2

    print("Successfully imported build_sam2")
except ImportError as e:
    print(f"Import error: {e}")

    # Print the Python path to see where Python is looking for modules
import sys

print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")