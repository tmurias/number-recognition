# number-recognition
Python machine learning project for classifying hand-written digits.

## Data
The sample data is in the `data/` directory. The handwritten digits (0-9) are 28x28 images stacked as 728x1 columns.
* `X1600.csv` (728x16000) contains 16000 training points, where the first 1600 columns are 0s, the next 1600 columns are 1s, etc.
* `Te28.csv` (728x10000) contains 10000 testing points
* `Lte28.csv` (10000x1) contains the labels (scalar, 0-9) corresponding to the test points

This data was provided by UVic's ECE 403 course.

## PCA
`pca.py` contains a Principal Component Analysis (PCA) algorithm for classification. Run `python3 pca.py` for results.

## Required modules
* `numpy`
* `scipy`
