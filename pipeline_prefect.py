"""
pipeline_prefect.py
-------------------
Orchestration du pipeline ML avec Prefect.

Flows disponibles : install, code, train, evaluate, predict, all

Exemples :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    predict_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"

# Seuls ces fichiers sont analysés (évite de scanner venv/ et autres dossiers)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
# Client d'exemple pour la prédiction (même exemple que le notebook)
SAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]


def run_command(cmd):
    """Exécute une commande, affiche sa sortie et lève une erreur si elle échoue."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Commande échouée (code {result.returncode}) : {cmd[2]}")


# ------------------------------------------------------------------ TASKS
@task(name="Installer les dépendances")
def install_dependencies():
    """Installe les paquets de requirements.txt."""
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code():
    """Formate les fichiers du projet avec black."""
    run_command([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="Qualité du code (flake8)")
def check_quality():
    """Analyse la qualité du code avec flake8."""
    run_command([sys.executable, "-m", "flake8", *PROJECT_FILES])


@task(name="Sécurité du code (bandit)")
def check_security():
    """Analyse de sécurité avec bandit (sévérité moyenne et plus)."""
    run_command([sys.executable, "-m", "bandit", "-ll", "-q", *PROJECT_FILES])


@task(name="Tests unitaires (pytest)")
def run_unit_tests():
    """Exécute les tests unitaires du dossier tests/."""
    run_command([sys.executable, "-m", "pytest", "tests", "-q"])


@task(name="Préparation des données")
def prepare_task(data_path=DATA_PATH):
    """Charge et prétraite les données."""
    return prepare_data(data_path)


@task(name="Entraînement du modèle")
def train_task(x_train, y_train):
    """Entraîne le modèle."""
    return train_model(x_train, y_train)


@task(name="Sauvegarde du modèle")
def save_task(model, path=MODEL_PATH):
    """Sauvegarde le modèle."""
    save_model(model, path)


@task(name="Chargement du modèle")
def load_task(path=MODEL_PATH):
    """Charge le modèle sauvegardé."""
    return load_model(path)


@task(name="Évaluation du modèle")
def evaluate_task(model, x_test, y_test):
    """Évalue le modèle."""
    return evaluate_model(model, x_test, y_test)


@task(name="Prédiction")
def predict_task(model, features=None):
    """Prédit le churn pour un client."""
    features = features or SAMPLE_CUSTOMER
    prediction = predict_model(model, features)
    print(f"Prédiction pour {features} : {'Churn' if prediction else 'Reste'}")
    return prediction


# ------------------------------------------------------------------ FLOWS
@flow(name="install")
def install_flow():
    """Flow : installation des dépendances."""
    install_dependencies()


@flow(name="code")
def code_flow():
    """Flow : formatage, qualité, sécurité et tests unitaires."""
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    """Flow : préparation des données + entraînement (+ sauvegarde)."""
    x_train, _, y_train, _ = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    """Flow : chargement du modèle + évaluation."""
    _, x_test, _, y_test = prepare_task()
    model = load_task()
    evaluate_task(model, x_test, y_test)


@flow(name="predict")
def predict_flow():
    """Flow : chargement du modèle + prédiction d'un client d'exemple."""
    model = load_task()
    predict_task(model)


@flow(name="all")
def all_flow():
    """Flow complet : code (format/qualité/sécurité/tests) puis préparation,
    entraînement, sauvegarde et évaluation."""
    format_code()
    check_quality()
    check_security()
    run_unit_tests()
    x_train, x_test, y_train, y_test = prepare_task()
    model = train_task(x_train, y_train)
    save_task(model)
    evaluate_task(model, x_test, y_test)


FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,  # alias (nom utilisé dans l'énoncé)
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "all": all_flow,
}


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
