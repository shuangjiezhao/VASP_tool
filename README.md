# VASP_tool: A Python Toolkit for VASP Data Analysis
VASP_tool is a Python library designed to streamline the analysis and visualization of data from VASP (Vienna Ab initio Simulation Package) calculations. It supports both band structure and density of states (DOS) computations with advanced customization options for publication-quality plots.

## Features
- Band Structure Analysis

-- Regular and spin-polarized band structure visualization.
-- Automatic handling of Fermi level shifts.
-- High symmetry k-point annotations.
-- Density of States (DOS) Analysis

-- Regular and projected DOS with support for spin polarization.
-- Smoothing and peak broadening using Savitzky-Golay filters and Gaussian functions.
-- Flexible orientations (vertical and horizontal plots).
-- Highly Customizable Plots

- Adjustable colors, labels, linewidths, and tick styles.
-- Publication-ready plotting with Matplotlib.

## Installation
```bash
1. Clone the repository:
git clone https://github.com/username/VASP_tool.git

2. Install the required Python dependencies:
pip install numpy pandas matplotlib scipy
```

## Usage
### Initialization
To start analyzing VASP data, create an instance of the VASP_tool class:
```python
from VASP_tool import VASP_tool

vasp = VASP_tool(
    files_path="path/to/vasp/output", 
    spin_polarization=True, 
    function="regular band", 
    fermi_level="VBM"
)
```
### Band Structure Plotting
```python
vasp.plot_regular_band(
    save_path="band_structure.png",
    figsize=(8, 6),
    band_color="blue",
    ylim=(-5, 5)
)
```

### DOS Plotting
```python
vasp.plot_regular_dos(
    save_path="dos.png",
    line_color="green",
    xlim=(-10, 10),
    ylim=(-5, 5)
)
```

### Projected DOS Plotting
```python
vasp.plot_projected_dos(
    spieces=["H", "O"],
    orbitals=["s", "p"],
    line_color=["blue", "red"]
)
```

## Input Requirements
- For Band Structure:
Files with names containing REFORMATTED, UP, DW, KLABELS, and KLINES.
- For DOS:
Files with names containing TDOS or specific species names for projected DOS.
Ensure files are named appropriately and placed in the same directory specified by files_path.

## Dependencies
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy

## Contributing
Contributions, suggestions, and bug reports are welcome. Please fork the repository and submit a pull request or open an issue.

## License
This project is licensed under the MIT License.
