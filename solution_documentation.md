# Hackathon Project: Facade Quality Control

## Project Overview

This project was developed as part of a **Hackathon**, with the goal of building an **AI-powered system for automated facade quality control**. The solution focuses on detecting and segmenting visual defects on building facades using computer vision and deep learning techniques.

The primary objective was to design an **end-to-end machine learning pipeline** capable of training, validating, and testing a robust model for facade defect detection, while maintaining clarity, scalability, and reproducibility.

---

## Team

* **Eni Magdalena Oreč**
* **Erna Topalović**
* **Filip Knapić**

---

## Problem Statement

Manual inspection of building facades is time-consuming, subjective, and prone to human error. The aim of this project is to:

* Automate the detection of facade defects
* Improve consistency and accuracy in quality control
* Enable scalable inspection using computer vision models

---

## Selected Approach

We implemented a **computer vision workflow based on instance segmentation**, enabling precise localization and classification of facade defects at the pixel level.

The chosen approach combines:

* **Roboflow** for dataset management and annotation
* **Python-based training and inference scripts**
* **YOLO11l** as the core deep learning model

---

## Workflow Architecture

The overall workflow consists of the following stages:

1. Data annotation and preparation in Roboflow
2. Dataset versioning and export
3. Model training using YOLO11l
4. Model evaluation on test data
5. Automated inference via Python scripts

This modular design allows easy iteration, retraining, and evaluation.

---

## Dataset Preparation

### Project Setup in Roboflow

* **Project Type:** Instance Segmentation
* **Objective:** Identify and segment all visible facade defects

Roboflow was used as the central platform for managing the dataset and annotations.

### Annotation Process

* Before annotation, a predefined set of defect classes was created in Roboflow
* These classes were used to consistently categorize all detected facade defects
* Multiple defect classes were theoretically possible at the beginning of the project
* However, across all training examples, only the classes listed below were observed and annotated

**Defined Defect Classes:**

* `brtva_ostecena`

* `defectssss`

* `lim_nedostaje`

* `staklo_puknuto`

* `vijak_labav`

* `vijak_nedostaje`

* All images were manually annotated

* Each visible defect on the facade was labeled according to the defined classes

* Pixel-accurate masks were created for instance segmentation

This ensured high-quality ground truth data for model training.

### Dataset Versioning and Export

* A finalized dataset version was generated
* The dataset was automatically split into:

  * **Training set**
  * **Validation set**
  * **Test set**
* The dataset was exported in a format compatible with YOLO-based training pipelines

---

## Model Training

### Training Framework

* **Model Architecture:** YOLO11l
* **Training Language:** Python

A custom Python script was developed to:

* Load the exported dataset
* Configure training parameters
* Train the YOLO11l model on the training set
* Validate performance using the validation set

This setup enabled efficient experimentation and performance tuning.

---

## Model Evaluation and Testing

A second Python script was implemented to handle inference and evaluation:

* Loads the trained YOLO11l model
* Takes the **test dataset** as input
* Runs inference on unseen images
* Outputs predictions for defect detection and segmentation

This separation between training and testing scripts ensures clarity and maintainability of the codebase.

---

## Key Technologies

* **Roboflow** – Dataset annotation, versioning, and export
* **Python** – Training and inference scripting
* **YOLO11l** – Deep learning model for instance segmentation

---

## Conclusion

The project successfully demonstrates a **complete AI-driven facade quality control pipeline**, from data annotation to model evaluation. By leveraging instance segmentation and modern deep learning architectures, the solution provides a strong foundation for scalable and automated facade inspection systems.

Future improvements may include:

* Expanding the dataset
* Fine-tuning model hyperparameters
* Integrating real-time or drone-based image acquisition

---

*Hackathon Project – Facade Quality Control*
