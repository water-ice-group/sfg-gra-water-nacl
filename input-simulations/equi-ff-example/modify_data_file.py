#!/usr/bin/env python

# remove bond and angle info from data file
with open('end_equil.data', 'r') as f:
        lines = f.readlines()
# Find the index of the line containing 'Bonds' and remove it and all lines after it
try:
    bonds_index = lines.index('Velocities\n')
    lines = lines[:bonds_index]
except ValueError:
    # 'Bonds' line not found in file
    pass
# Remove lines containing 'bonds', 'angles', or 'dihedrals'
lines = [line for line in lines if not any(word in line for word in ['bonds', 'angles','bond', 'angle'])]

pair_coeffs_index = None
atoms_index = None
for i, line in enumerate(lines):
    if line.startswith('Pair Coeffs'):
        pair_coeffs_index = i
    elif line.startswith('Atoms'):
        atoms_index = i
        break
lines = lines[:pair_coeffs_index] + lines[atoms_index:]

with open('end_equil_for_nnp.data', 'w') as f:
        f.writelines(lines)