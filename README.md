# Color Extraction for Dual-color AR Waveguide Display for Color Vision Assistance


## Overview

This repository provides the artificial neural network (ANN)-based color tracking algorithm used to identify red-color regions in real-world scenes. The extracted color regions serve as overlay cues in a dual-color holographic optical element (CM-HOE)-based augmented-reality (AR) waveguide display, designed to assist users with color vision deficiency (CVD) in discriminating red-green color pairs.

The algorithm operates on the HSV color space and was implemented in Python with a feed-forward ANN. It is also accompanied by a fast gradient sign method (FGSM) robustness evaluation and a CVD perception simulation pipeline based on standard linear-transform models.

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

```
python train_ann.py
```

The trained model weights will be saved in the `models/` folder.

### 2. Run color extraction

To run the trained ANN on a new input image:

```
python color_extraction.py --input path/to/image.png --output path/to/output_mask.png
```

The output is a binary mask indicating regions detected as red.

### 3. Run CVD simulation

To simulate how a CVD observer would perceive an input image:

```
python cvd_simulation.py --input path/to/image.png --type protanopia --output path/to/simulated.png
```

Available simulation types: `protanopia`, `deuteranopia`, `tritanopia`.

## Data Description

The dataset (`dataset/`) consists of 1,000 annotated images of red and non-red objects captured under varied illumination conditions, including indoor, outdoor, low-light, and shadow-dominant scenes. Each image is paired with a binary mask indicating ground-truth red-region annotations.

Data partitioning used in the manuscript:
- Training set: 700 images (70 %)
- Validation set: 150 images (15 %)
- Test set: 150 images (15 %)

A fixed random seed was used for partitioning to ensure reproducibility.

## Citation

If you use this code or data in your research, please cite our paper:

```
Cho, S.-H., Baek, D.-H., Choi, W. J., & Choi, Y.-W. (2026).
Dual-color augmented-reality waveguide display for color vision assistance using color tracking.
iScience (under review).
```

## Contact

For questions or further information, please contact the lead contact:

**Young-Wan Choi, Ph.D.**
Department of Intelligent Semiconductor Engineering
School of Electrical and Electronic Engineering
Chung-Ang University
84 Heukseok-ro, Dongjak-gu, Seoul 06974, Republic of Korea
Email: ychoi@cau.ac.kr

## License

This repository is provided for academic and research purposes. Please contact the lead contact for other uses.
