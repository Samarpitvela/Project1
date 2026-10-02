"""
House Price Prediction (ML project)

Dataset: California Housing (1990 census), from the book "Hands-On Machine Learning"
by A. Geron. Each row is one district (block group) of California.
We predict median_house_value (in US dollars).

Run:  python house_price.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/housing.csv")
print("Rows:", len(df))
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string())

# --- Problem 1: prices are capped at 500,001 in this dataset -> remove capped rows
capped = (df["median_house_value"] >= 500001).sum()
df = df[df["median_house_value"] < 500001].copy()
print(f"Removed {capped} rows where the price was capped at $500,001")

# --- Problem 2: total_rooms is for the whole district, not one house -> make per-household features
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["people_per_household"] = df["population"] / df["households"]

TARGET = "median_house_value"
NUMERIC = ["longitude", "latitude", "housing_median_age", "median_income",
           "rooms_per_household", "bedrooms_per_room", "people_per_household", "households"]
CATEGORICAL = ["ocean_proximity"]          # location as text

X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Problem 3: total_bedrooms has missing values -> fill with median inside the pipeline
prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42),
}

rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rows.append({"model": name, "RMSE ($)": round(rmse), "R2": round(r2_score(y_test, pred), 3)})
    if name == "Random Forest":
        names = [n.split("__")[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
        imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)

print(f"\nTrain rows: {len(X_train)}   Test rows: {len(X_test)}")
print(pd.DataFrame(rows).to_string(index=False))
print("\nTop features (Random Forest):")
print(imp.head(5).round(3).to_string())
print(f"\nAverage price in test set: ${y_test.mean():,.0f}")
