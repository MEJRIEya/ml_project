"""
deploiement_prefect.py
----------------------
Déploie les flows Prefect avec un planning (worker = ce script).

Terminal 1 : prefect server start
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        # Pipeline complet : tous les jours à 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        # Entraînement : tous les jours à 03:00
        train_flow.to_deployment(
            name="ml-pipeline-train",
            cron="0 3 * * *",
            tags=["training", "mlops"],
        ),
        # Les autres sont déclenchés manuellement
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["code", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        predict_flow.to_deployment(name="ml-pipeline-predict", tags=["predict", "mlops"]),
    )
