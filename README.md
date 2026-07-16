# License Plate Similarity Verification using Siamese Neural Networks

## Overview

This project presents a deep metric learning approach for license plate similarity verification using a Siamese Neural Network. Rather than recognizing the characters on a license plate, the proposed system determines whether two vehicle images contain the same license plate.

The system utilizes a ResNet18 backbone with shared weights to extract feature embeddings from two input images. These embeddings are compared using cosine similarity to predict whether both images belong to the same license plate.

A web-based interface developed using Streamlit allows users to upload two vehicle images and obtain real-time similarity predictions.

---

## Features

- Siamese Neural Network for image similarity verification
- ResNet18 feature extraction backbone
- Shared-weight architecture
- 128-dimensional feature embeddings
- L2 Normalization
- Cosine Similarity comparison
- Contrastive Loss training
- Online Hard Negative Mining (OHNM)
- Streamlit web application
- Real-time license plate verification

---

## Dataset

This project uses the **UFPR-ALPR Dataset**, a publicly available Automatic License Plate Recognition dataset collected under real-world driving conditions.

Dataset includes:

- 4,500 vehicle images
- 150 unique vehicles
- Cars, motorcycles, buses, and trucks
- Multiple images captured under different viewpoints and lighting conditions
- Images collected from moving vehicles

Dataset Link:

https://web.inf.ufpr.br/vri/databases/ufpr-alpr/

---

## Project Structure

```
COSC880/
│
├── app.py                  # Streamlit web application
├── model.py                # Siamese Network architecture
├── best_model.pth          # Trained model weights
├── requirements.txt        # Python dependencies
├── README.md
│
├── Dataset/
│
├── Downloads/
│
├── models/
│
└── LicensePlateProject/
```

---

## Requirements

- Python 3.10 or later
- pip
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your_username>/<repository_name>.git
```

Move into the project directory:

```bash
cd <repository_name>
```

(Optional) Create a virtual environment:

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run app.py
```

The application will open automatically in your browser.

---

## Usage

1. Open the Streamlit application.
2. Upload two vehicle images.
3. Click **Verify License Plates**.
4. The system computes feature embeddings for both images.
5. Cosine similarity is calculated.
6. The prediction is displayed as:

- Same License Plate
- Different License Plate

along with the similarity score.

---

## Model Architecture

The proposed model consists of:

- Two identical ResNet18 branches
- Shared weights
- Projection Head
- 128-dimensional embeddings
- L2 Normalization
- Cosine Similarity

Training uses:

- Contrastive Loss
- Online Hard Negative Mining (OHNM)
- Adam Optimizer

---

## Technologies Used

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision |
| Backbone Network | ResNet18 |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Machine Learning Utilities | Scikit-learn |
| User Interface | Streamlit |
| Development Environment | Jupyter Notebook |

---

## Performance

The proposed method achieved high verification performance using the Siamese Neural Network with Online Hard Negative Mining.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cosine Similarity

---

## Author

**Fatima Malik**

Master of Science in Computer Science

Towson University

Advisor:

Dr. Akshita Maradapu Vera Venkata Sai

---

## License

This project was developed for academic and research purposes as part of the COSC 880 Master's Capstone Project at Towson University.
