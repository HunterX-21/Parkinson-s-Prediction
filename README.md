# AI Diabetes Detection

An interactive machine learning web application that predicts the likelihood of diabetes based on user-provided health measurements.

The application uses a Random Forest Classifier trained on a diabetes dataset and provides an interactive interface built with Streamlit.

## Overview

This project demonstrates how machine learning can be integrated into a simple web application to perform binary classification.

Users can enter health-related measurements through an interactive sidebar, and the trained machine learning model generates a diabetes classification.

The application also provides basic information about the dataset, including a data table, descriptive statistics, and a visualization.

## Features

- Interactive Streamlit web interface
- Diabetes prediction using a Random Forest Classifier
- User input through interactive sliders
- Dataset preview and descriptive statistics
- Dataset visualization
- Model test accuracy display
- Real-time prediction based on user input

## Technologies Used

- **Python**
- **Pandas** – Data manipulation and analysis
- **Scikit-learn** – Machine learning
- **Random Forest Classifier** – Classification model
- **Streamlit** – Interactive web application
- **Pillow (PIL)** – Image handling

## Input Features

The model uses the following eight features:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Glucose level |
| Blood Pressure | Blood pressure measurement |
| Skin Thickness | Skin thickness measurement |
| Insulin | Insulin level |
| BMI | Body Mass Index |
| DPF | Diabetes Pedigree Function |
| Age | Age of the individual |

## Machine Learning Approach

The application loads the diabetes dataset using Pandas and separates the dataset into:

- **Features (X):** The first eight columns
- **Target (Y):** The final column

The dataset is divided into training and testing sets using a 75/25 split.

A Random Forest Classifier is then trained using the training data and evaluated using the test data.

The application displays the resulting test accuracy and uses the trained model to generate a prediction for the user's input.

## Project Structure

```text
AI-Diabetes-Detection/
│
├── webml.py
├── diabetes.csv
├── headerB.jpg
├── requirements.txt
├── README.md
└── .gitignore
