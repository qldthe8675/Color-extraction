# Color extraction for dual-color AR Waveguide display for color vision assistance


## Overview

This repository provides the artificial neural network (ANN)-based color tracking algorithm used to identify red-color regions in real-world scenes. The extracted color regions serve as overlay cues in a dual-color holographic optical element (CM-HOE)-based augmented-reality (AR) waveguide display, designed to assist users with color vision deficiency (CVD) in discriminating red-green color pairs.

The algorithm operates on the HSV color space and was implemented in Python with a feed-forward ANN.

<img src="./ANN architecture for color extraction.png" width="50%"
 height="50%">
<img src="./Masked images.png" width="50%"
 height="50%">
## Repository Contents

- `AI_HSV_color extraction.py` — Main script for ANN-based red-color region extraction.
- `training_images.zip` — CVD samples data set for traning and evaluation.

## Requirements

- Python 3.13.3
- OpenCV
- NumPy
- Pandas
- Colour
- Matplotlib
- SciPy

You can install all dependencies via:

```
pip install opencv-python numpy pandas colour-science matplotlib scipy
```


## How to Use

### 1. Train the ANN (optional)

If you wish to retrain the network from scratch using the provided dataset:


### 2. Run color extraction

To run the trained ANN on a new input image:

The output is a binary mask indicating regions detected as red.


## Data Description

The dataset (`dataset/`) consists of 1,000 annotated images of red and non-red objects captured under varied illumination conditions, including indoor, outdoor, low-light, and shadow-dominant scenes. Each image is paired with a binary mask indicating ground-truth red-region annotations.

Data partitioning used in the manuscript:
- Training set: 700 images (70 %)
- Validation set: 150 images (15 %)
- Test set: 150 images (15 %)

A fixed random seed was used for partitioning to ensure reproducibility.


## Contact

For questions or further information, please contact the lead contact in the manuscript:
