# Mammography AI: Core Preprocessing Pipeline

A professional computer vision framework designed to prepare raw mammography scans for deep learning architectures. This repository focuses on the critical first steps of medical image analysis: standardizing files, enhancing anatomical features, and isolating the Region of Interest (ROI).

## Project Purpose
The accuracy of medical AI models heavily depends on the quality of input data. This project provides an automated pipeline to:
- Convert and scale raw 16-bit DICOM images into standardized 8-bit formats.
- Enhance microcalcifications and hidden lesions using frequency-domain analysis (Wavelet Transforms).
- Automatically crop and center breast tissue (Laterality-Aware) to remove irrelevant background noise.

## Repository Structure
    medical-ai-core/
    ├── dataset/                 # (Local only) Data storage
    │   ├── raw_dicom/           # Place your raw .dcm files here
    │   └── processed/           # Output directory for tensor-ready PNGs
    ├── preprocessing/           # Core logic (Wavelet Transform, Smart Crop)
    ├── tests/                   # Validation & Visualization dashboards
    ├── dicom_processor.py       # Main engine for batch-converting DICOM files
    ├── config.py                # Centralized configuration (paths, parameters)
    └── requirements.txt         # Project dependencies

## How to Use

### 1. Setup the Environment
Install the core data science dependencies:
    pip install numpy opencv-python PyWavelets matplotlib pydicom dicom2jpg

Install PyTorch with CUDA 12.4 support for GPU acceleration:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

### 2. Prepare the Data
Place your raw .dcm (DICOM) files into the dataset/raw_dicom/ directory. (Note: This directory is git-ignored to protect sensitive medical data).

### 3. Run the Pipeline
First, execute the processor to convert raw DICOMs into standard images:
    python dicom_processor.py

Then, launch the visual validation dashboard to inspect the edge-boosting and cropping results:
    python tests/test_advanced_prep.py

## Future Work (To-Do)
-  Model Architecture: Design and implement advanced CNN backbones (e.g., ResNet, EfficientNet, or Swin Transformer) for classification.
-  Training Loop: Build a robust PyTorch training pipeline with CUDA acceleration.
- Clinical Metrics: Implement medical evaluation metrics (AUC-ROC, Sensitivity, Specificity) to validate model performance.
'@

Set-Content -Path README.md -Value $content -Encoding UTF8
git add .
git commit -m "docs: add comprehensive README with structure, usage steps, and future roadmap"
git push