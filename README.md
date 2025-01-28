# BirdDatasetNN

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Application](#Application)
  - [Training notebook](#Training-notebook)
- [File Structure](#file-structure)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Alexanderms36/BirdDatasetNN.git
   cd BirdDatasetNN
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure the following libraries are included in your `requirements.txt`:
   - `python 3.10.13`
   - `pytorch 2.5.1`
   - `matplotlib 3.8.4`
   - `seaborn 0.13.2`
   - `scikit-learn 1.4.2`
   - `torchvision 0.20.1`
   - `pillow 10.3.0`

## Usage

### Application

To launch the app you can run `app.py` file:

```bash
py app.py
```

### Training notebook

Model can be retrained in `train.ipynb`. Choose the kernel and run the blocks in a notebook.

## File Structure
 - `app_test_images`: Folder contains images to test the application (`app.py`)
 - `bird_dataset`: 
 - `app.py`: Application that detects bird class. It takes an image path and return predicted class.
 - `load.py`: Python module for loading a data. If `is_dataset` flag is true it transforms directory into a dataset. (else it takes just one image).
 - `model_weights.pth`: Model parameters you can use to load a model.
 - `model.py`: Module contains a model class.
 - `requirements.txt`: List of dependencies required to run the project
 - `train.ipynb`: Jupyter notebook contains a model training process.