# ChurnGuard AI

AI-powered customer retention intelligence system

## Overview

ChurnGuard AI is a Streamlit-based web application designed to predict customer churn and support proactive retention planning. It helps businesses identify customers who are likely to leave and provides actionable insights that can be used to reduce churn before it happens.

Customer churn prediction matters because retaining existing customers is often more cost-effective than acquiring new ones. By identifying risk early, businesses can focus their retention efforts on the customers who need attention most and improve overall customer lifetime value.

## Problem Statement

Businesses often struggle with customer retention because churn signals are not always obvious until it is too late. Traditional approaches can be reactive, making it difficult to recognize early warning signs such as short tenure, pricing concerns, support gaps, or contract flexibility.

This project addresses the need for an early-warning system that helps teams detect churn risk in advance and prioritize retention actions more effectively.

## Solution

ChurnGuard AI combines machine learning with a business-focused user experience. The application accepts customer profile information, runs it through a trained churn prediction model, and classifies the customer as Low Risk, Medium Risk, or High Risk based on predicted churn probability.

In addition to prediction, the system generates customer-specific reasons and tailored retention recommendations so that businesses can act on the output with clarity and confidence.

## Key Features

- Secure user authentication with login and account creation
- Customer churn prediction workflow
- Probability-based churn risk scoring
- Low / Medium / High risk classification
- Dynamic churn-driver analysis based on customer profile
- Personalized retention recommendations
- Prediction history tracking for logged-in users
- Analytics dashboard with KPI-style metrics
- Data visualization for risk and churn patterns
- CSV export/download for analytics summaries
- About section with project and model information

## Tech Stack

The application is built using the following technologies:

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- SQLite
- Pickle
- Matplotlib (used in the project notebook and related analysis)
- Jupyter Notebook

## Machine Learning Approach

The project uses the Telco Customer Churn dataset stored in the dataset folder. The workflow is as follows:

1. Load and preprocess the dataset
2. Convert categorical inputs into model-compatible feature representations
3. Train a Random Forest Classifier on the churn dataset
4. Load the trained model from the serialized artifact
5. Predict churn probability for new customer profiles
6. Convert the probability into a business-friendly risk level
7. Generate customer-specific insights and actions

The implementation uses a train/test split during evaluation to assess the model on unseen data.

## Model Performance

The current tuned model used by the application reports:

- Accuracy: 75.09%
- Precision: 52.10%
- Recall: 76.20%
- F1 Score: 61.89%

## Application Workflow

User Input
↓
Data Preprocessing
↓
ML Model Prediction
↓
Risk Analysis
↓
Customer Insights
↓
Retention Recommendation

## Project Structure

```text
ChurnGuard-AI/
├── app.py
├── auth.py
├── database.py
├── dashboard.py
├── customer_churn_model.pkl
├── churn_model.pkl
├── users.db
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/
├── models/
├── notebooks/
│   ├── Customer_Churn_Prediction.ipynb
│   └── scaler.pkl
├── reports/
└── README.md
```

## Getting Started

1. Clone the repository
2. Install the required Python packages:
   - streamlit
   - pandas
   - scikit-learn
   - numpy
   - matplotlib
3. Run the application:

```bash
streamlit run app.py
```

## Usage

- Sign up or log in to access the app
- Enter customer details in the prediction interface
- Review the churn probability and risk label
- Explore the generated insights and recommended actions
- Use the analytics dashboard to review churn trends and key metrics
- Export summary reports when needed

## Notes

This project is designed as a practical AI product prototype for churn prediction and customer retention strategy, with a polished interface suitable for demos, hackathons, and portfolio presentations.
