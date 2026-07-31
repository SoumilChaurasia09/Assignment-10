import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_dataset():
    """
    Downloads/loads the Heart Disease Dataset and preprocesses it.
    """
    dataset_path = 'heart.csv'
    
    # Check if local heart.csv exists, otherwise download from UCI repository
    if not os.path.exists(dataset_path):
        print("Downloading Heart Disease Dataset from UCI repository...")
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, header=None, names=cols, na_values='?')
        
        # Handle missing values using mode
        df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
        df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
        
        # Convert target to binary (0: No Disease, 1: Heart Disease Detected)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Format chest pain (cp) to 0-3 scale standard in Kaggle datasets if needed
        # (1-4 -> 0-3)
        df['cp'] = df['cp'].astype(int) - 1
        
        # Ensure correct datatypes
        int_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal', 'target']
        for col in int_cols:
            df[col] = df[col].astype(int)
            
        df.to_csv(dataset_path, index=False)
        print(f"Dataset successfully saved to {dataset_path}")
    else:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset from {dataset_path}")
        
    return df

def main():
    print("=" * 60)
    print("TASK 1: DATA UNDERSTANDING & PREPROCESSING")
    print("=" * 60)
    
    # 1. Load dataset
    df = prepare_dataset()
    
    # 2. Display first 5 records
    print("\n[1] First 5 Records of the Dataset:")
    print(df.head())
    
    # 3. Identify Numerical Features & Target Variable
    target_variable = 'target'
    features = [col for col in df.columns if col != target_variable]
    print(f"\n[2] Features ({len(features)}): {features}")
    print(f"    Target Variable: {target_variable}")
    print(f"    Target Distribution:\n{df['target'].value_counts().to_dict()}")
    
    # 4. Check missing values
    missing_values = df.isnull().sum()
    print("\n[3] Missing Values Check:")
    print(missing_values)
    print(f"Total Missing Values: {missing_values.sum()}")
    
    # 5. Train / Test Split (80% Train, 20% Test)
    X = df[features]
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[4] Data Split Completed:")
    print(f"    Training Set Shape: {X_train.shape}")
    print(f"    Testing Set Shape:  {X_test.shape}")
    
    print("\n" + "=" * 60)
    print("TASK 2: MODEL DEVELOPMENT & EVALUATION")
    print("=" * 60)
    
    # Compare candidate algorithms
    models = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Support Vector Machine (SVM)': SVC(probability=True, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model: {name:<30} | Test Accuracy: {acc * 100:.2f}%")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"\n---> Selected Best Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")
    
    # Save best model to model.pkl
    model_filename = 'model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"---> Saved trained model artifact to {model_filename}")
    
    # Verify load
    loaded_model = joblib.load(model_filename)
    test_sample = X_test.iloc[0:1]
    sample_pred = loaded_model.predict(test_sample)[0]
    print(f"---> Verification load check on sample: Prediction = {sample_pred}")

if __name__ == '__main__':
    main()
