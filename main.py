import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import shap
import joblib
import warnings

# Suppress minor warnings for a cleaner output console
warnings.filterwarnings('ignore')

def load_data():
    """Loads the training and testing datasets."""
    print("Loading datasets...")
    train_df = pd.read_csv('fraudTrain.csv')
    test_df = pd.read_csv('fraudTest.csv')
    return train_df, test_df

def preprocess(df, fit_scaler=True, scaler=None):
    """Cleans data, encodes text to numbers, and scales numerical values."""
    df = df.copy()

    # Drop columns that don't help the AI find patterns
    cols_to_drop = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'dob', 'unix_time', 'merchant', 'street', 'city', 'state', 'zip']
    df = df.drop(cols_to_drop, axis=1, errors='ignore')

    # Convert text categories into numbers (AI requires numbers)
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Standardize the size of numbers so large values don't overpower small ones
    num_cols = df.select_dtypes(include=['float64', 'int64']).drop('is_fraud', axis=1, errors='ignore').columns
    if fit_scaler:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df, scaler
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        return df

def train_model(X_train, y_train):
    """Trains the XGBoost AI model using balanced data."""
    print("\nTraining the AI Model... (This might take a moment)")
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def human_readable_evaluation(model, X_test, y_test):
    """Evaluates the model and prints a report that anyone can understand."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Break down the confusion matrix into simple terms
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    auc_score = roc_auc_score(y_test, y_prob)

    # --- PLAIN ENGLISH REPORT ---
    print("\n" + "="*55)
    print(" 📊 PLAIN ENGLISH FRAUD DETECTION REPORT")
    print("="*55)
    print(f"Total transactions analyzed: {len(y_test):,}")
    
    print("\n✅ THE SUCCESSES:")
    print(f" - Normal transactions safely approved: {tn:,}")
    print(f" - Fraudulent transactions successfully caught: {tp:,}")
    
    print("\n⚠️ THE MISTAKES:")
    print(f" - False Alarms (Real customers temporarily blocked): {fp:,}")
    print(f" - Missed Fraud (Scammers who slipped through): {fn:,}")
    
    print("\n🎯 OVERALL AI GRADE:")
    print(f" - Accuracy Score (ROC-AUC): {auc_score * 100:.1f}%")
    if auc_score > 0.90:
        print("   (Excellent! The AI is very good at separating fraud from normal activity.)")
    elif auc_score > 0.80:
        print("   (Good, but could be improved to reduce false alarms or missed frauds.)")
    else:
        print("   (Needs work. The AI is struggling to identify fraud accurately.)")
    print("="*55 + "\n")

    # --- SIMPLIFIED ROC CURVE ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AI Performance ({auc_score * 100:.1f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (50%)')
    
    # Using plain English labels for the graph
    plt.xlabel('False Alarm Rate (Innocent transactions flagged)')
    plt.ylabel('Catch Rate (Actual frauds blocked)')
    plt.title('How Well the AI Separates Fraud vs. Legitimate Transactions')
    plt.legend(loc="lower right")
    plt.show()

def explain_model_simply(model, X_sample):
    """Generates a simple bar chart showing what data the AI looks at most."""
    print("\n🧠 GENERATING AI EXPLANATION...")
    print("Displaying the 'Risk Factors' Bar Chart.")
    print("This shows which transaction details (like amount, category, etc.)")
    print("were most important in helping the AI make its decisions.")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # The simple bar plot you requested earlier
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Top Risk Factors for Fraud")
    plt.show()

if __name__ == "__main__":
    # 1. Load Data
    train_df, test_df = load_data()

    # 2. Preprocess Training Data
    train_df_clean, scaler = preprocess(train_df, fit_scaler=True)
    X_train = train_df_clean.drop('is_fraud', axis=1)
    y_train = train_df_clean['is_fraud']

    # 3. Balance the Data (Give the AI enough examples of fraud to learn from)
    print("Balancing data using SMOTE to ensure fair training...")
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # 4. Train Model
    model = train_model(X_resampled, y_resampled)

    # 5. Preprocess Test Data (Using the exact same rules as the training data)
    test_df_clean = preprocess(test_df, fit_scaler=False, scaler=scaler)
    X_test = test_df_clean.drop('is_fraud', axis=1)
    y_test = test_df_clean['is_fraud']

    # 6. Evaluate and Translate Results to Plain English
    human_readable_evaluation(model, X_test, y_test)

    # 7. Visual Explainability (Bar Chart)
    explain_model_simply(model, X_test.sample(100, random_state=42))

    # 8. Save the final pipeline
    print("\nSaving the AI model and data scaler for future use...")
    joblib.dump(model, 'fraud_detection_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Process Complete!")