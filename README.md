# Cityscapes Semantic Segmentation using Frozen DINOv2 Representations and Task-Specific Segmentation Head

- **Author:** Dhani Crapels
- **Codalab Username:** Dhani
- **Codalab Competition:** [The 5LSM0 competition (2025)](https://codalab.lisn.upsaclay.fr/competitions/21622)
- **TU/e Email Address:** d.r.m.crapels@student.tue.nl

## Overview
This repository contains the code and materials for implementing a robust semantic segmentation model using frozen DINOv2 representations and a task-specific segmentation head. The model is designed to segment urban scenes from the **Cityscapes dataset** and is aimed at achieving high robustness across varying environmental conditions.

This project was developed as part of the course **5LSM0: Neural Networks for Computer Vision**, offered by the Department of Electrical Engineering at **Eindhoven University of Technology**.

## Getting Started

### 1. Clone the Repository
To get started with this project, you need to clone the repository to your local machine or a SLURM Cluster. You can do this by running the following command in your terminal:
```bash
git clone https:https://github.com/Dhani-AI/NNCV
```

After cloning the repository, navigate to the project directory:
```bash
cd "Final assignment"
```

(Refer to ``README-Installation.md`` for detailed instructions on cloning and potentially setting up your environment and tools using a High-Performance Computing (HPC) cluster).

### 2. SLURM Cluster Environment Setup

The code downloads all the necessary training and validation data, alongside with a Docker container that contains all the necessary modules, on a Huggingface page as explained in ``README-Slurm.md``. 

### 3. Local installation

You can can also obtain the dataset from the [official Cityscapes website](https://www.cityscapes-dataset.com/) for local experiments. Once downloaded, make sure to organize the dataset as follows:

```plaintext
data/
├── cityscapes/
│   ├── leftImg8bit/
│       ├── train/
│           ├── aachen/
│           ├── .../
│       ├── val/
│           ├── .../
│   ├── gtFine/
│       ├── train/
│           ├── aachen/
│           ├── .../
│       ├── val/
│           ├── .../
```

To install the required libraries, simply run the following command in your terminal:

```bash
pip install -r requirements.txt
```

*Note: It is recommended setting up a virtual environment to keep the dependencies organized. Also install PyTorch first, matching the system's CUDA version if available.* 

## File Descriptions

- `train.py`: Script for training the DINOv2 backbone with added segmentation head.
- `dino_model.py`: Contains the `DINOv2SegmentationModel` class, which loads pretrained DINOv2 features and allows switching between segmentation heads (e.g., linear, FPN).
- `unet_model.py`: Lightweight implementation of a U-Net model used for comparison experiments without pretrained features.
- `main.sh`: Bash script executed by SLURM (via `jobscript_slurm.sh`) to run the training process inside the container. Sets up a temporary venv.
- `jobscript_slurm.sh`: SLURM submission script that requests compute resources and runs `main.sh`. Intended for use on HPC clusters with SLURM job scheduling.
- `download_docker_and_data.sh`: SLURM-compatible script for downloading the Apptainer `.sif` container and the Cityscapes dataset from Hugging Face.
- `process_data.py`: Preprocessing script used for generating submission files for Codalab evaluation. Applies the same normalization and resizing steps used during training.
- `config_vits14.py`: Configuration file for loading and initializing DINOv2-Small (ViT-S/14) backbone necessary for the `dino_model.py`.
- `config_vitb14.py`: onfiguration file for loading and initializing DINOv2-Base (ViT-B/14) backbone necessary for the `dino_model.py`.
- `config_vitl14.py`: Configuration file for loading and initializing DINOv2-Larfe (ViT-L/14) backbone necessary for the `dino_model.py`.
- `config_vitg14.py`: Configuration file for loading and initializing DINOv2-Giant (ViT-G/14) backbone necessary for the `dino_model.py`.
- `.env`: Environment file storing sensitive or system-specific variables, such as Weight and Biases API tokens.