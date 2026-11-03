# Healthcare Budget Overrun Prediction

## 1. Project Overview

This project aims to predict future healthcare expenses using historical spending data and a Linear Regression model.

The objective is to help healthcare insurance managers anticipate future expenses and improve budget planning.

**Current scope:** Sprint 2 focuses exclusively on healthcare expense prediction for the next three months.

## 2. Sprint 2 Objectives

* Analyze historical healthcare expenses.
* Select relevant features and target variables.
* Prepare the dataset for machine learning.
* Perform a chronological train/test split.
* Develop and train a Linear Regression model.
* Predict healthcare expenses for the next three months.
* Evaluate model performance using MAE and RMSE.
* Visualize historical and predicted expenses.

## 3. Technologies

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit
* Git and GitHub
* Jira

## 4. Project Structure

```text
healthcare-budget-prediction/
│
├── data/
│   └── healthcare_expenses.csv
│
├── src/
│   └── model.py
│
├── app.py
├── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Model Description

The project uses Linear Regression to predict healthcare expenses based on historical monthly spending.

* **Input feature (X):** month_number
* **Target variable (y):** expenses
* **Training/testing strategy:** chronological split (8 training months and 4 testing months).
* **Evaluation metrics:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* **Forecast horizon:** the next three months.

The final forecasting model is trained using all available historical observations.

## 6. Interactive Application

The project includes a Streamlit application that allows users to:

* Enter monthly healthcare expenses.
* Generate predictions for the next three months.
* View model evaluation metrics (MAE and RMSE).
* Visualize historical and predicted expenses.

### Run the application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

## 7. Agile Organization

**Epic:** Healthcare Budget Overrun Prediction

**Sprint 2:** Healthcare Expense Prediction

**User Story:** Develop and evaluate a Linear Regression model to predict future healthcare expenses using historical data.

### Sprint 2 Tasks

* HEAL-11: Variable Selection
* HEAL-15: Future Expenses Prediction
* HEAL-25: Prediction Visualization

## 8. Project Status

**Sprint 2: Implemented and tested.**

The current implementation uses synthetic demonstration data.

The model and application are designed as an initial prototype. Real-world healthcare expenditure data will be required for meaningful operational predictions.
