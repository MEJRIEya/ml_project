"""main.py : exécute les étapes du pipeline via des arguments CLI."""

import argparse
import sys

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description="Pipeline ML - Customer Churn")
    parser.add_argument(
        "--step", choices=["prepare", "train", "evaluate", "all"], default="all"
    )
    parser.add_argument("--data", default="Churn_Modelling.csv")
    parser.add_argument("--model", default="classifier.joblib")
    args = parser.parse_args()

    try:
        x_train, x_test, y_train, y_test = prepare_data(args.data)

        if args.step == "prepare":
            print(f"Données prêtes : train={x_train.shape}, test={x_test.shape}")
        elif args.step == "train":
            save_model(train_model(x_train, y_train), args.model)
        elif args.step == "evaluate":
            evaluate_model(load_model(args.model), x_test, y_test)
        else:
            model = train_model(x_train, y_train)
            evaluate_model(model, x_test, y_test)
            save_model(model, args.model)
    except (FileNotFoundError, ValueError) as err:
        print(f"Erreur : {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
