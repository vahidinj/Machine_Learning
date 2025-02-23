import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# OBJECTIVE: Linear regression to determine CO2 emissions based on multiple vehicle features

# Data Loading
data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(data_url)

# Data Familiarization/Visualization
df.sample()
df.describe()
df.info()

# Dropping irrelevant columns
df = df.drop(
    [
        "MODELYEAR",
        "MAKE",
        "MODEL",
        "VEHICLECLASS",
        "TRANSMISSION",
        "FUELTYPE",
    ],
    axis=1,
)
df.sample(5)

# Correlation matrix and heatmap
df.corr()
sns.heatmap(df.corr())
# plt.show()

# Dropping columns to avoid multicollinearity
df = df.drop(
    [
        "CYLINDERS",
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB",
    ],
    axis=1,
)
df.head()

# Pairplot visualization
sns.pairplot(data=df, diag_kws={"kde": True})
# plt.show()

# MODEL BUILDING
X = df.drop("CO2EMISSIONS", axis=1).to_numpy()
y = df["CO2EMISSIONS"].to_numpy()

# Standardizing the features
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X=X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, random_state=42
)

# Training the linear regression model
mlt_reg = linear_model.LinearRegression().fit(X_train, y_train)

# Extracting model coefficients and intercept
std_coefs_ = mlt_reg.coef_
std_intercept_ = mlt_reg.intercept_
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

print(f"Standardized Coefficients: {std_coefs_}")
print(f"Standardized Intercept: {std_intercept_}")

# MODEL EVALUATIONS
predictions = mlt_reg.predict(X_test)
print(f"Predicted CO2 Emissions: {predictions}")
print(f"Actual CO2 Emissions: {y_test}")

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2}")

# Converting coefficients and intercept back to original scale
orig_coefs = std_coefs_ / std_devs_
orig_intercept = std_intercept_ - np.sum((means_ * std_coefs_) / std_devs_)

print(
    f"Original Coefficients are: {orig_coefs} and the Original Intercept is: {orig_intercept}"
)

# Prediction equation
print(f"The y_pred equation is as follows: {X @ orig_coefs + orig_intercept}")

# Model Visualization
# Disclaimer: This is only a cross-section visualization ... actual visualization is a 3D graph.

# Visualization with original values
plt.figure()
plt.scatter(X_train[:, 0], y_train, color="blue")
plt.plot(X_train[:, 0], orig_coefs[0] * X_train[:, 0] + orig_intercept, "-r")
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.show()

# Visualization with standardized values
plt.figure()
plt.scatter(X_train[:, 0], y_train, color="blue")
plt.plot(X_train[:, 0], std_coefs_[0] * X_train[:, 0] + std_intercept_, "-r")
plt.xlabel("Standardized Engine size")
plt.ylabel("Standardized CO2 Emissions")
plt.show()


plt.show()


# NOTICE: The Standardized regression looks much better.
# This is why when/if using the model on new data and making predictions one should standardize the new features.

# Example with new data
new_data = np.array([[2, 33]])  # Example new data

# Standardize the new data using the same scaler
new_data_std = std_scaler.transform(new_data)

# Make predictions using the standardized new data
new_predictions_std = mlt_reg.predict(new_data_std)

print(f"Predicted CO2 Emissions (Standardized): {new_predictions_std}")
