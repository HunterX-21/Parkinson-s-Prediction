# Parkinson's Disease Prediction 🧠

An interactive machine learning web application that predicts whether a person may have Parkinson's disease based on biomedical voice measurement features.

The application is built with Python, Scikit-learn, Pandas, and Streamlit, and provides predictions using two machine learning classification algorithms:

- Random Forest Classifier
- Logistic Regression

## 🚀 Live Demo

👉 **[Launch the Parkinson's Prediction App](https://parkinson-s-prediction-qyjlry8ujdedsdmtcm83dr.streamlit.app/)**

## 📌 Project Overview

Parkinson's disease is a progressive neurological disorder that can affect movement, speech, and other functions.

This project demonstrates how machine learning can be applied to voice-related biomedical measurements to classify data into two categories:

- **0 — Negative**
- **1 — Positive**

The application allows users to interactively enter values for different voice measurements through the Streamlit sidebar and receive predictions from the trained machine learning models.

> **Disclaimer:** This project is an educational machine learning demonstration and is not intended to provide medical diagnosis or replace professional medical advice.

## 🛠️ Technologies Used

- **Python**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Scikit-learn** — Machine learning
- **Pillow (PIL)** — Image handling
- **Streamlit** — Interactive web application
- **Git & GitHub** — Version control and project hosting

## 🤖 Machine Learning Models

Two classification algorithms are implemented in the application.

### Random Forest Classifier

The Random Forest model is trained using the project's Parkinson's dataset and used to predict the user's input.

The application also calculates the model's accuracy using the test dataset.

### Logistic Regression

A Logistic Regression model is also trained and evaluated on the same dataset.

The implementation uses:

- `C = 0.4`
- `max_iter = 1000`
- `solver = liblinear`

The accuracy of the Logistic Regression model is displayed alongside the Random Forest results.

## 📊 Dataset

The project uses `parkinsons.csv`.

The dataset contains biomedical voice measurements associated with Parkinson's disease classification.

The following feature categories are used by the application:

- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- MDVP:Jitter(%)
- MDVP:Jitter(Abs)
- MDVP:RAP
- MDVP:PPQ
- Jitter:DDP
- MDVP:Shimmer
- MDVP:Shimmer(dB)
- Shimmer:APQ3
- Shimmer:APQ5
- MDVP:APQ
- Shimmer:DDA
- NHR
- HNR
- RPDE
- DFA
- spread1
- spread2
- D2
- PPE

The `name` column is excluded from the model features, while `status` is used as the target variable.

## ⚙️ How It Works

The application follows these general steps:

1. Load the Parkinson's dataset.
2. Display information and statistics about the dataset.
3. Separate the features from the target variable.
4. Split the data into training and testing sets.
5. Train a Random Forest Classifier.
6. Train a Logistic Regression model.
7. Accept user-entered biomedical measurements.
8. Generate predictions using both models.
9. Display the classification results and model accuracy.

## 📁 Project Structure

```text
Parkinson-s-Prediction/
│
├── webml.py
├── parkinsons.csv
├── headerB.jpg
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
