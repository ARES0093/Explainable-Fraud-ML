# Explainable-Fraud-ML
A transparent machine learning pipeline using XGBoost and SHAP to detect credit card fraud, featuring human-readable evaluation reports and clear visual insights.


# 🛡️ Explainable-Fraud-ML

## 1. Project Overview
**Explainable-Fraud-ML** is an end-to-end machine learning pipeline designed to detect fraudulent credit card transactions. Unlike traditional models that act as a "black box," this project prioritizes transparency. It not only achieves high accuracy in identifying fraud but also translates its predictive logic into plain English and simple visual charts, bridging the gap between complex data science and business operations.

## 2. The Problem 
In the financial sector, detecting credit card fraud is a massive challenge due to highly imbalanced datasets (millions of normal transactions vs. a handful of frauds). Furthermore, when an AI model denies a customer's transaction, businesses face a "Black Box" problem: stakeholders and customer support teams cannot explain *why* the transaction was blocked. Lack of explainability leads to a lack of trust in the AI system.

## 3. The Solution
This project solves both issues by building a highly accurate detection model while making its decision-making process fully transparent. 
* It uses **SMOTE** to handle the imbalanced data, ensuring the AI has enough examples of fraud to learn from. 
* It uses **SHAP (SHapley Additive exPlanations)** to break down the model's logic, visually ranking the exact transactional attributes (like amount, location, or time) that influenced the fraud flag.

## 4. Technical Approach
* **Data Preprocessing:** Unnecessary metadata is stripped, categorical variables are converted into a machine-readable format using `LabelEncoder`, and numerical features are standardized using `StandardScaler`.
* **Class Balancing:** `SMOTE` (Synthetic Minority Over-sampling Technique) is applied exclusively to the training data to generate synthetic fraud examples, preventing the model from biasing toward the majority class.
* **Modeling:** An `XGBoost` classifier is trained using a logistic objective function, optimized for binary classification.
* **Evaluation:** Traditional metrics (ROC-AUC, Confusion Matrix) are extracted and automatically translated into a human-readable console report.

## 5. Key Features
* **High-Accuracy Detection:** Robust classification of legitimate vs. fraudulent transactions using XGBoost.
* **Human-Readable Reporting:** Generates a plain-English terminal report explicitly stating "The Successes" (frauds caught) and "The Mistakes" (false alarms).
* **Transparent Visual Insights:** Outputs a clean SHAP bar chart highlighting the top risk factors driving the AI's predictions.
* **Simplified ROC Curve:** Provides an accessible performance graph with plain-English axis labels (e.g., "False Alarm Rate").
* **Ready for Production:** Automatically saves the trained model and data scaler as `.pkl` files for future deployment.

## 6. Project Workflow
1. **Data Loading:** Imports raw training and testing datasets.
2. **Preprocessing:** Cleans data, encodes text, and scales numeric values.
3. **Balancing:** Applies SMOTE to ensure a fair training environment.
4. **Training:** Fits the XGBoost model to the balanced data.
5. **Evaluation:** Tests the model on unseen data and prints a plain-English grade.
6. **Explainability:** Generates visual SHAP summaries of the AI's logic.
7. **Serialization:** Exports the model and scaler via Joblib.

## 7. Tech Stack
* **Language:** Python 3.x
* **Core ML Library:** Scikit-Learn
* **Algorithm:** XGBoost (`xgboost`)
* **Data Balancing:** Imbalanced-Learn (`imblearn`)
* **Explainability:** SHAP (`shap`)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Serialization:** Joblib

## 8. Setup and Installation

**Step 1: Clone the repository (or set up your project folder)**
Ensure your `main.py` file, `fraudTrain.csv`, and `fraudTest.csv` are all in the same directory.

**Step 2: Create a Virtual Environment (Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Installation
pip install pandas scikit-learn xgboost imbalanced-learn shap matplotlib joblib

#Run 
python main.py
```

## 9. Future Improvements
* **Web Dashboard Integration: Wrap the model in a Streamlit or Flask web app to allow non-technical users to manually input transaction details and get a real-time fraud probability score.

* **Hyperparameter Tuning: Implement GridSearchCV or Optuna to squeeze out even higher ROC-AUC performance while keeping false alarms to a minimum.

* **Real-Time Data Pipeline: Adapt the preprocessing functions to handle streaming data (e.g., via Kafka) for live transaction monitoring.
