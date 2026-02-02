# tests/test_transformers.py

import numpy as np
import pandas as pd

from src.pipeline import AlignColumns, Cleaner, FeatureBuilder


def test_align_columns_adds_missing_and_drops_extra():
    X = pd.DataFrame(
        {
            "a": [1, 2],
            "extra": ["x", "y"],
        }
    )
    tr = AlignColumns(expected_columns=["a", "b"])
    Xt = tr.transform(X)

    assert list(Xt.columns) == ["a", "b"]
    assert Xt["a"].tolist() == [1, 2]
    assert Xt["b"].isna().all()


def test_cleaner_totalcharges_numeric_and_fill_is_stable():
    X = pd.DataFrame(
        {
            "tenure": [1, 2, None],
            "TotalCharges": [" 10 ", "20", " "],  # last becomes NaN after to_numeric
            "Contract": ["Month-to-month", "Two year", None],
        }
    )
    cl = Cleaner()
    Xt = cl.fit_transform(X)

    # TotalCharges must become numeric and filled (median of [10,20] = 15)
    assert pd.api.types.is_numeric_dtype(Xt["TotalCharges"])
    assert float(Xt.loc[2, "TotalCharges"]) == 15.0

    # tenure filled with median of [1,2] = 1.5
    assert pd.api.types.is_numeric_dtype(Xt["tenure"])
    assert np.isclose(float(Xt.loc[2, "tenure"]), 1.5)

    # Contract missing filled with mode (Month-to-month or Two year; mode depends on pandas tie-break)
    assert Xt["Contract"].isna().sum() == 0


def test_feature_builder_is_robust_to_missing_columns():
    # Missing many fields: should not crash and should create engineered columns with defaults.
    X = pd.DataFrame(
        {
            "Contract": [None, "Month-to-month"],
            # no tenure / no MonthlyCharges / no TotalCharges
        }
    )
    fb = FeatureBuilder().fit(X)
    Xt = fb.transform(X)

    expected = [
        "avg_monthly_bill",
        "high_charges",
        "is_new_client",
        "is_long_term",
        "tenure_bucket",
        "num_services",
        "is_month_to_month",
        "is_auto_pay",
        "revenue_per_tenure",
    ]
    for col in expected:
        assert col in Xt.columns

    # with missing tenure/monthly/total -> safe defaults
    assert (Xt["avg_monthly_bill"] == 0).all()
    assert (Xt["revenue_per_tenure"] == 0).all()
    assert (Xt["high_charges"] == 0).all()
