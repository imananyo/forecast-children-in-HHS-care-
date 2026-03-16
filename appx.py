import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="HHS Shelter Forecast", layout="wide")

st.title("HHS Shelter Demand Forecast")
st.write("Predictive Forecasting of Care Load & Placement Demand")

# -------------------------------
# Load and Clean Data
# -------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    df = df.dropna(subset=["Date"])

    df["Date"] = pd.to_datetime(df["Date"])

    # clean numeric columns
    cols = [
        "Children in HHS Care",
        "Children transferred out of CBP custody",
        "Children discharged from HHS Care"
    ]

    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df = df.sort_values("Date")

    df = df.set_index("Date")

    return df


df = load_data()

# -------------------------------
# Feature Engineering
# -------------------------------
df["care_lag1"] = df["Children in HHS Care"].shift(1)
df["care_lag7"] = df["Children in HHS Care"].shift(7)
df["care_lag14"] = df["Children in HHS Care"].shift(14)

df["rolling_mean_7"] = df["Children in HHS Care"].rolling(7).mean()
df["rolling_mean_14"] = df["Children in HHS Care"].rolling(14).mean()

df["net_pressure"] = (
    df["Children transferred out of CBP custody"]
    - df["Children discharged from HHS Care"]
)

df["pressure_7day"] = df["net_pressure"].rolling(7).sum()

df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

df = df.dropna()

# -------------------------------
# Train/Test Split
# -------------------------------
train_size = int(len(df) * 0.8)

train = df[:train_size]
test = df[train_size:]

features = [
    "care_lag1",
    "care_lag7",
    "care_lag14",
    "rolling_mean_7",
    "rolling_mean_14",
    "net_pressure",
    "pressure_7day",
    "day_of_week",
    "month"
]

target = "Children in HHS Care"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# -------------------------------
# Train Random Forest Forecast Model
# -------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

test["forecast"] = rf.predict(X_test)

# -------------------------------
# Capacity Breach Classifier
# -------------------------------
capacity_limit = st.slider(
    "Shelter Capacity Limit",
    min_value=5000,
    max_value=15000,
    value=10000
)

df["capacity_breach"] = (
    df["Children in HHS Care"] > capacity_limit
).astype(int)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

clf.fit(X_train, train["capacity_breach"])

test["breach_probability"] = clf.predict_proba(X_test)[:, 1]

# -------------------------------
# Dashboard Metrics
# -------------------------------
current_population = int(df["Children in HHS Care"].iloc[-1])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Current Children in HHS Care",
        current_population
    )

with col2:
    risk = test["breach_probability"].iloc[-1]
    st.metric(
        "Capacity Breach Probability",
        round(risk, 2)
    )

# -------------------------------
# Population Forecast Chart
# -------------------------------
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    test.index,
    test["Children in HHS Care"],
    label="Actual"
)

ax.plot(
    test.index,
    test["forecast"],
    label="Forecast"
)

ax.axhline(
    capacity_limit,
    color="red",
    linestyle="--",
    label="Capacity Limit"
)

ax.set_title("Shelter Population Forecast")
ax.legend()

st.pyplot(fig)

# -------------------------------
# Risk Indicator
# -------------------------------
if risk > 0.7:
    st.error("High Risk of Capacity Breach")

elif risk > 0.4:
    st.warning("Moderate Risk of Capacity Breach")

else:
    st.success("Shelter Capacity Stable")

    