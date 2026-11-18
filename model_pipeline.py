"""model_pipeline.py : fonctions modulaires du pipeline ML (churn client)."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]
FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def prepare_data(csv_path, test_size=0.2, random_state=1):
    """Charge, prétraite et sépare les données. Retourne x_train, x_test, y_train, y_test."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être compris entre 0 et 1.")

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET}' absente du fichier.")

    df = df.drop_duplicates()
    if df.isna().any().any():
        df = df.dropna()

    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et le retourne."""
    if len(x_train) != len(y_train):
        raise ValueError("x_train et y_train n'ont pas la même taille.")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle, affiche les métriques et retourne un dictionnaire."""
    y_pred = model.predict(x_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
    print(f"Accuracy : {results['accuracy'] * 100:.2f}%")
    print("Matrice de confusion :")
    print(results["confusion_matrix"])
    print("Rapport de classification :")
    print(results["report"])
    return results


def predict_model(model, features):
    """
    Prédit le churn pour un client.

    Args:
        model: modèle entraîné.
        features (list): 9 valeurs dans l'ordre de FEATURE_COLUMNS
            (CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
            HasCrCard, IsActiveMember, EstimatedSalary).

    Returns:
        int: 1 si le client risque de partir, 0 sinon.
    """
    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"{len(FEATURE_COLUMNS)} valeurs attendues : {FEATURE_COLUMNS}"
        )
    x = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    return int(model.predict(x)[0])


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
