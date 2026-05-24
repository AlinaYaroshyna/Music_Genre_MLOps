# Music Genre Classification: End-to-End MLOps

This isn't just another model in a notebook. This project is a full-scale MLOps pipeline designed to take raw audio, extract meaningful features, and serve predictions through a production-ready API. It bridges the gap between signal processing research and scalable engineering.

---

# The Mission

Manual genre tagging is slow and subjective. This system automates the process using the GTZAN dataset, transforming raw .wav files into structured data to power recommendation engines or automated playlist curation.

## Key Capabilities

- Signal Processing: Automated feature extraction (MFCCs, Spectrograms) using Librosa
- Experiment Tracking: Every hyperparameter and metric is logged via MLflow
- Reproducibility: Fully containerized environment with Docker
- CI/CD: Automated testing and build cycles via GitHub Actions

---

# Tech Stack

| Category | Tools |
| :--- | :--- |
| Core ML | Python, Scikit-learn, Librosa, Pandas, NumPy |
| MLOps | MLflow, DVC, Docker |
| Deployment | FastAPI, GitHub Actions |

---

# Project Structure

Music_Genre_MLOps/
```
├── src/
│   ├── data_processing/      # Signal cleaning & normalization
│   ├── training/             # Model training & hyperparameter tuning
│   └── inference/            # API logic for real-time predictions
│
├── api/                      # FastAPI implementation
├── notebooks/                # Research & EDA
├── tests/                    # Pytest suite
├── Dockerfile                # Containerization
├── requirements.txt          # Dependencies
└── README.md
```
---

# Quick Start

## 1. Clone the Repository

`git clone https://github.com/AlinaYaroshyna/Music_Genre_MLOps.git`

`cd Music_Genre_MLOps`

## 2. Create Virtual Environment

### Linux / macOS

```
python -m venv venv

source venv/bin/activate
```
### Windows
```
python -m venv venv

venv\Scripts\activate
```

## 3. Install Dependencies

pip install -r requirements.txt

---

# Run the Pipeline

## Train the Model

`python train.py`

Training metrics and hyperparameters are automatically logged to MLflow.

## Run Predictions

`python predict.py`

---

# Deploy with Docker

## Build Docker Image

`docker build -t music-genre-mlops .`

## Run Container

`docker run -p 8000:8000 music-genre-mlops`

---

# API Usage

Once the container is running, send a POST request to /predict.

## Request
```
{
  "audio_file": "path/to/song.wav"
}
```
## Response
```
{
  "predicted_genre": "Jazz",
  "confidence": 0.92
}
```
---

# Portfolio Highlights

This project demonstrates a transition from Data Scientist to ML Engineer by focusing on:

- Modular Code Architecture
- Automated Testing
- Experiment Reproducibility
- Deployment-Oriented Engineering

---

# Author

Alina Yaroshyna

GitHub: https://github.com/AlinaYaroshyna

LinkedIn: www.linkedin.com/in/yaroshyna-alina

---

# Support

If you find this project helpful, feel free to star the repository and share feedback!
