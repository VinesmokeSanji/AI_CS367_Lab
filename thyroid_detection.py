import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def get_dataset(file_url):
    cols = ["Target", "Feat1", "Feat2", "Feat3", "Feat4", "Feat5"]
    data_frame = pd.read_csv(file_url, header=None, names=cols, na_values='?')
    return data_frame
  

def build_and_train_model(X_train_scaled, y_train):
    hist_gb_model = HistGradientBoostingClassifier()
    hist_gb_model.fit(X_train_scaled, y_train)
    joblib.dump(hist_gb_model, 'saved_thyroid_model.pkl')
    return hist_gb_model
  

def prepare_data(data_frame):
    features = data_frame.drop(columns="Target")
    target = data_frame["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def assess_model(hist_gb_model, X_test_scaled, y_test):
    y_pred = hist_gb_model.predict(X_test_scaled)
    
    print("Model Performance Report:\n")
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n")
    print(conf_matrix)

if __name__ == "__main__":
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
    
    dataset = get_dataset(data_url)
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(dataset)
    
    trained_model = build_and_train_model(X_train_scaled, y_train)
    
    assess_model(trained_model, X_test_scaled, y_test)
