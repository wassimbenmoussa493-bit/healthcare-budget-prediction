import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Healthcare Expenses Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Expenses Prediction")
st.subheader("Sprint 2 : Prédiction des dépenses de santé")

st.write(
    "Saisissez les dépenses des 12 derniers mois "
    "pour prédire automatiquement les 3 prochains mois."
)

# Valeurs initiales proposées
default_expenses = [
    12000, 12500, 13000, 12800,
    13500, 14000, 13800, 14500,
    15000, 14800, 15500, 16000
]

months = [
    "Janvier", "Février", "Mars", "Avril",
    "Mai", "Juin", "Juillet", "Août",
    "Septembre", "Octobre", "Novembre", "Décembre"
]

st.header("1. Saisir les dépenses mensuelles")

expenses = []

with st.form("expenses_form"):

    columns = st.columns(3)

    for i, month in enumerate(months):

        with columns[i % 3]:
            value = st.number_input(
                f"{month} (DT)",
                min_value=0.0,
                value=float(default_expenses[i]),
                step=500.0
            )

            expenses.append(value)

    submitted = st.form_submit_button(
        "Générer les prédictions"
    )

if submitted:

    if sum(expenses) == 0:
        st.error("Veuillez saisir des dépenses valides.")
        st.stop()

    # Préparer les données
    df = pd.DataFrame({
        "month_number": range(1, 13),
        "expenses": expenses
    })

    X = df[["month_number"]]
    y = df["expenses"]

    # 2. Séparation chronologique : 8 mois / 4 mois
    X_train = X.iloc[:8]
    X_test = X.iloc[8:]

    y_train = y.iloc[:8]
    y_test = y.iloc[8:]

    # 3. Entraîner le modèle pour l'évaluation
    evaluation_model = LinearRegression()
    evaluation_model.fit(X_train, y_train)

    # 4. Prédictions sur les 4 derniers mois
    y_pred = evaluation_model.predict(X_test)

    # 5. Évaluer le modèle
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    st.header("2. Évaluation du modèle")

    col1, col2 = st.columns(2)

    col1.metric("MAE", f"{mae:,.2f} DT")
    col2.metric("RMSE", f"{rmse:,.2f} DT")

    # 6. Entraîner le modèle final sur les 12 mois
    final_model = LinearRegression()
    final_model.fit(X, y)

    # 7. Prédire les 3 prochains mois
    future_months = pd.DataFrame({
        "month_number": [13, 14, 15]
    })

    predictions = final_model.predict(future_months)

    st.header("3. Prévisions des 3 prochains mois")

    future_names = [
        "Janvier 2026",
        "Février 2026",
        "Mars 2026"
    ]

    cols = st.columns(3)

    for i, col in enumerate(cols):
        col.metric(
            future_names[i],
            f"{predictions[i]:,.2f} DT"
        )

    # 8. Visualisation
    st.header("4. Visualisation des dépenses")

    historical_months = list(range(1, 13))
    predicted_months = [13, 14, 15]

    plt.figure(figsize=(12, 5))

    plt.plot(
        historical_months,
        expenses,
        marker="o",
        label="Dépenses historiques"
    )

    plt.plot(
        predicted_months,
        predictions,
        marker="o",
        linestyle="--",
        label="Dépenses prédites"
    )

    plt.title("Healthcare Expenses: Historical Data & Predictions")
    plt.xlabel("Mois")
    plt.ylabel("Dépenses (DT)")
    plt.xticks(
        range(1, 16),
        months + ["Janvier", "Février", "Mars"],
        rotation=30
    )

    plt.legend()
    plt.grid(True)

    st.pyplot(plt)

    st.success("Prédictions générées avec succès !")