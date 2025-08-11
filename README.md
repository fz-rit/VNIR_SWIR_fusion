# VNIR–SWIR Hyperspectral Fusion Toolkit

Brief tools to:
- Coregister a SWIR cube to a VNIR cube via GUI-selected control points.
- Build a full-spectrum VNIR+SWIR cube with weighted blending in the overlap region.

Repo:
- [coregister_controlpoints_gui.py](coregister_controlpoints_gui.py): interactive coregistration (homography at ~950 nm).
- [build_cube.py](build_cube.py): merges registered VNIR+SWIR cubes into one ENVI cube.

## Installation

- Python 3.9 or 3.10;
- Create and activate a new conda environment;
- Install dependencies: `pip install numpy opencv-python matplotlib spectral scipy`

## Usage

1) Coregister SWIR to VNIR
- Open [coregister_controlpoints_gui.py](coregister_controlpoints_gui.py) and edit the `vnir_path` and `swir_path` variables at the bottom of the file to point to your ENVI headers (.hdr).
- Run: `python coregister_controlpoints_gui.py`
- In the GUI:
  - Right-click to add at least 4 corresponding points in each image (zoom via toolbar magnifier if needed).
  - Press Esc to accept (or close the figure to retry).
- Output: saves SWIR as `<original_swir>_warped.hdr` next to the input.
- Notes:
  - Assumes SWIR is horizontally flipped vs VNIR (handled via np.fliplr).
  - Uses averaged bands near 950 nm to compute a homography for the warp.

2) Build full-spectrum cube
- Ensure you have a registered pair: VNIR and SWIR_warped on the same spatial grid.
- Open [build_cube.py](build_cube.py) and set:
  - `infolder`, `vnir_path_dat` (base path without .hdr), `swir_path_dat` (base path of warped SWIR without .hdr),
  - `outfolder`, `full_outfilehdr` (output .hdr path), and `saveimage = 1`.
- Run: `python build_cube.py`
- What it does:
  - Reads wavelengths from headers, finds VNIR–SWIR overlap,
  - Resamples SWIR overlap to VNIR wavelengths and blends with linear weights,
  - Concatenates VNIR + blended overlap + remaining SWIR, sorts wavelengths,
  - Writes ENVI uint16 cube with metadata including `reflectance scale factor = 10000`.

## Requirements and Assumptions

- ENVI headers must contain valid `wavelength` metadata for both cubes.
- A GUI backend for Matplotlib is required to pick control points.
- For large cubes, ensure sufficient RAM.

## Citation

If this toolkit is helpful, please cite:

Plain text:
- Messinger D., Macalintal J., Hassanzadeh A., VNIR–SWIR Fusion Toolkit (2023–2024). Software, version X.Y.

BibTeX:
@misc{vnir_swir_fusion_toolkit,
  title   = {VNIR--SWIR Fusion Toolkit},
  author  = {Messinger, David and Macalintal, J. and Hassanzadeh, Amir},
  year    = {2023},
  note    = {Software, version X.Y},
  howpublished = {Git repository}
}

Also consider citing the libraries used: Spectral Python (spectral), OpenCV, NumPy.