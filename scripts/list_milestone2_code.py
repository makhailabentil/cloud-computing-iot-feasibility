#!/usr/bin/env python3
"""List all new code files added for Milestone 2."""

from pathlib import Path

milestone2_files = {
    'src/data_processing/capture24_loader.py': 'CAPTURE-24 Data Loader',
    'src/evaluation/systematic_evaluation.py': 'Systematic Evaluation Framework',
    'src/evaluation/__init__.py': 'Evaluation Package Init',
    'scripts/evaluate_capture24.py': 'Main Evaluation Script',
    'scripts/evaluate_all_participants.py': 'Full Evaluation Script',
    'scripts/get_real_capture24_data.py': 'Data Preparation Script',
    'scripts/download_capture24.py': 'Download Helper Script',
    'scripts/prepare_capture24_data.py': 'Data Preparation Helper',
    'scripts/compare_results.py': 'Results Comparison Script',
    'scripts/get_detailed_stats.py': 'Statistics Script',
}

print("=" * 70)
print("MILESTONE 2: NEW CODE FILES")
print("=" * 70)
print()

total_lines = 0
existing_files = []

for filepath, description in milestone2_files.items():
    path = Path(filepath)
    if path.exists():
        lines = len(open(path, encoding='utf-8', errors='ignore').readlines())
        total_lines += lines
        existing_files.append((filepath, description, lines))
        print(f"[+] {filepath}")
        print(f"  Description: {description}")
        print(f"  Lines of Code: {lines}")
        print()

print("=" * 70)
print(f"TOTAL: {len(existing_files)} files, {total_lines} lines of code")
print("=" * 70)

print("\n=== CODE BREAKDOWN BY MODULE ===\n")

# Core modules
core_modules = [f for f in existing_files if f[0].startswith('src/')]
scripts = [f for f in existing_files if f[0].startswith('scripts/')]

print("Core Modules (src/):")
core_lines = 0
for filepath, desc, lines in core_modules:
    print(f"  - {filepath}: {lines} lines")
    core_lines += lines
print(f"  Subtotal: {core_lines} lines\n")

print("Scripts (scripts/):")
script_lines = 0
for filepath, desc, lines in scripts:
    print(f"  - {filepath}: {lines} lines")
    script_lines += lines
print(f"  Subtotal: {script_lines} lines\n")

print(f"GRAND TOTAL: {total_lines} lines of new code for Milestone 2")

