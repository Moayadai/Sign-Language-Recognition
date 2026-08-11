import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    data_dir = "data"
    csv_path = os.path.join(data_dir, "dataset.csv")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sign_language_model.pkl")

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please run data_collection.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Dataset is empty.")
        return
        
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
