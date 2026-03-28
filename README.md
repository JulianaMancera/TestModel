# TestModel – Driver Monitoring System 
&emsp; &emsp; This repository contains the implementation of DMS‑HybridNet, a hybrid deep learning architecture designed for real‑time driver monitoring under challenging conditions such as low‑light environments and occlusion. The system leverages CNNs, BiLSTMs, and Attention mechanisms to efficiently detect driver states, with deployment support via TensorFlow Lite for mobile devices.

## ⚠️Note on Model Usage
This repository currently includes an *experimental implementation of DMS‑HybridNet (CNN + BiLSTM + Attention).*
Please be aware that:

- This is *not the final approach* we will use in our project.
- The model here is only a *test run* to check feasibility with my **own dataset.**
- The scripts and dataset are meant for **practice and validation**, not for production deployment.

## Model Architecture
DMS‑HybridNet combines three key components:

- **CNN (Convolutional Neural Networks)** → Extracts spatial features from driver facial images.
- **BiLSTM (Bidirectional Long Short‑Term Memory)** → Captures temporal dependencies across video frames.
- **Attention Mechanism** → Focuses on the most relevant features (e.g., drowsiness cues).

## Repository Structure
### Data Handling
- `collect_data.py` → Collects raw driver data (camera/sensor input).
- `clean_data.py` → Cleans and preprocesses datasets.
- `extract_features.py` → Extracts features for training.
### Testing & Deployment
- `test_camera.py` → Tests camera input.
- `test_setup.py` → Verifies environment setup.
- `test_tflite.py` → Runs TensorFlow Lite model tests.
- `live_test.py` → Real‑time driver monitoring demo.
### Datasets
- `driver_dataset.csv` → Raw dataset.
- `driver_dataset_clean.csv` → Cleaned dataset.

## Usage Flow
- **Data Collection** → Run `collect_data.py`
- **Data Cleaning** → Run `clean_data.py`
- **Feature Extraction** → Run `extract_features.py`
- **Model Training** → Train DMS‑HybridNet with extracted features
- **Testing & Deployment** → Use `test_tflite.py` and `live_test.py` for real‑time monitoring

## Installation
git clone `https://github.com/JulianaMancera/TestModel.git`
cd `TestModel`
python `-m venv venv`
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install `-r requirements.txt`

## Applications
- Real‑time driver drowsiness detection
- Monitoring under low‑light conditions
- Mobile deployment using TensorFlow Lite
