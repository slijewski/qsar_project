import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    print("Loading fingerprints...")
    try:
        df = pd.read_csv('data/egfr_fingerprints.csv')
    except FileNotFoundError:
        print("Error: data/egfr_fingerprints.csv not found. Run 02_eda_descriptors.py first.")
        exit(1)
    
    # Handle missing values
    print("Checking for missing values...")
    if df.pIC50.isnull().any():
        print(f"Dropping {df.pIC50.isnull().sum()} rows with missing pIC50 values.")
        df = df.dropna(subset=['pIC50'])
    
    X = df.drop('pIC50', axis=1)
    y = df.pIC50
    
    print(f"Input data: {X.shape}, Target: {y.shape}")
    
    # Remove low variance features
    print("Removing low variance features...")
    selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
    X = selection.fit_transform(X)
    print(f"Data after dimensionality reduction: {X.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Model performance on test set:")
    print(f"R2 Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Save outputs
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    joblib.dump(model, 'outputs/egfr_model.pkl')
    joblib.dump(selection, 'outputs/variance_selection.pkl')
    print("Model and selector saved.")
    
    # Plot
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red') # Ideal line
    plt.title(f'Prediction vs Reality (R2={r2:.2f})')
    plt.savefig('outputs/model_performance.png')
    print("Saved plot outputs/model_performance.png")
