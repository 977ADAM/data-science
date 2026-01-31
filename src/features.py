import pandas as pd

"""
def build_features(df):
    df = df.copy()

    df["avg_monthly_bill"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_long_term"] = (df["tenure"] > 24).astype(int)
    df["num_services"] = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
    )

    return df
"""

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========
    # Денежные
    # =========
    df["avg_monthly_bill"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["high_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # =========
    # Время жизни
    # =========
    df["is_new_client"] = (df["tenure"] < 12).astype(int)
    df["is_long_term"] = (df["tenure"] > 24).astype(int)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # =========
    # Продукты
    # =========
    services = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["num_services"] = sum((df[s] == "Yes").astype(int) for s in services)

    # =========
    # Контракт и оплата
    # =========
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    # =========
    # Доходность клиента (маркетинг ⭐)
    # =========
    df["revenue_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df
