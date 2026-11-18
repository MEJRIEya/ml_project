"""Tests unitaires de model_pipeline.py."""

import pandas as pd
import pytest

from model_pipeline import (
    FEATURE_COLUMNS,
    predict_model,
    prepare_data,
    train_model,
)


def test_prepare_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        prepare_data("inexistant.csv")


def test_prepare_data_split_shapes():
    x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert list(x_train.columns) == FEATURE_COLUMNS


def test_train_model_size_mismatch():
    with pytest.raises(ValueError):
        train_model([[1], [2]], [0])


def test_predict_model_returns_binary():
    x = pd.DataFrame([[0] * 9, [1] * 9], columns=FEATURE_COLUMNS)
    model = train_model(x, [0, 1], n_estimators=5)
    assert predict_model(model, [0] * 9) in (0, 1)


def test_predict_model_wrong_length():
    with pytest.raises(ValueError):
        predict_model(None, [1, 2, 3])
