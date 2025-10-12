#!/usr/bin/env python
"""Preprocess code samples for GNN vulnerability detection."""

from pathlib import Path

# Input and output files
input_file = Path("data/preprocessed/diversevul/diversevul/diversevul_20230702.json")
output_file = Path("data/preprocessed/diversevul/diversevul/diversevul.json")

# Read lines from the input file
with input_file.open("r") as f:
    lines = f.readlines()

# Strip whitespace and add comma to each line
lines = [line.strip() + "," for line in lines if line.strip()]

# Wrap lines with brackets
lines.insert(0, "[")
lines.append("]")

# Write to output file
with output_file.open("w") as f:
    f.write("\n".join(lines))

print(f"Formatted file saved as '{output_file}'")

input_file.unlink()
print(f"Removed original file '{input_file}'")
