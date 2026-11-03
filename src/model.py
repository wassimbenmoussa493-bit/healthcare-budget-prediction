import pandas as pd
from pathlib import Path

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the dataset path
DATA_PATH = BASE_DIR / "data" / "healthcare_expenses.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display the data
print("Healthcare Expenses Dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()
# Convert month column to datetime
df["month"] = pd.to_datetime(df["month"])

# Create a numerical feature for the month
df["month_number"] = range(1, len(df) + 1)

# Display the prepared dataset
print("\nPrepared Dataset:")
print(df[["month", "month_number", "expenses"]])
# HEAL-11: Selection of features and target variable
X = df[["month_number"]]
y = df["expenses"]

# Split the data chronologically
X_train = X.iloc[:8]
X_test = X.iloc[8:]

y_train = y.iloc[:8]
y_test = y.iloc[8:]

# Display the split results
print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
from sklearn.linear_model import LinearRegression
# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Display model parameters
print("\nModel Training Completed!")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
# Predict expenses on test data
y_pred = model.predict(X_test)

# Display actual vs predicted values
print("\nActual vs Predicted Expenses:")

for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f} DT | Predicted: {predicted:.2f} DT")
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

# Display evaluation results
print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
import matplotlib.pyplot as plt
# Visualize actual vs predicted expenses

plt.figure(figsize=(10, 5))

plt.plot(y_test.values, marker="o", label="Actual Expenses")
plt.plot(y_pred, marker="o", label="Predicted Expenses")

plt.title("Actual vs Predicted Healthcare Expenses")
plt.xlabel("Test Months")
plt.ylabel("Expenses (DT)")

plt.legend()
plt.grid(True)

plt.show()
# Predict expenses for the next 3 months

future_months = pd.DataFrame({
    "month_number": [13, 14, 15]
})

future_predictions = model.predict(future_months)

print("\nFuture Expenses Predictions:")

for month, prediction in zip(
    ["January 2026", "February 2026", "March 2026"],
    future_predictions
):
    print(f"{month}: {prediction:.2f} DT")