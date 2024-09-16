# Patch4Breast
Cut patches and get 3-classes lables for breast cancer

## Core Requirements
- sdpc-for-python
- PIL == 0.9.50

## Matters Needing Attention
- Patch_1evel represents the number of downsampling times, and the specific cutting magnification depends on three factors: initial magnification, pyramid downsampling zoom, and patch_1evel. For example, the initial magnification is 40X, zoom=0.5, and when patch_1evel=1, it is 20X. When it equals 2, it is 10X.
If zoom=0.25, it is 5X when patch_1evel=1. When it equals 2, it is 5X. Currently, the code supports zoom=0.5
