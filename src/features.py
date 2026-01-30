def build_features(df):
    df = df.copy()

    df["avg_monthly_bill"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_long_term"] = (df["tenure"] > 24).astype(int)
    df["num_services"] = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
    )

    return df
