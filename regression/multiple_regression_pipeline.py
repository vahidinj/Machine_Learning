import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# OBJECTIVE: Linear regression to determine CO2 emissions based on multiple vehicle features using a pipeline


# Data Loading
data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(data_url)

# Data Familiarization/Visualization (Optional)
df.describe()
sns.heatmap(df.corr())
sns.pairplot(data=df, diag_kws={"kde": True})

# Data Preprocessing
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

df = df.drop(
    [
        "CYLINDERS",
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB",
    ],
    axis=1,
)

# MODEL BUILDING
X = df.drop("CO2EMISSIONS", axis=1)  # No need for to_numpy() with Pipeline
y = df["CO2EMISSIONS"]

# Splitting the data *before* scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the pipeline
pipeline = Pipeline(
    [
        ("scaler", preprocessing.StandardScaler()),
        ("regressor", linear_model.LinearRegression()),
    ]
)

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# MODEL EVALUATIONS
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2}")

# Coefficient and Intercept Extraction (from the pipeline)
linear_model_part = pipeline.named_steps["regressor"]
scaler_part = pipeline.named_steps["scaler"]

std_coefs_ = linear_model_part.coef_
std_intercept_ = linear_model_part.intercept_
means_ = scaler_part.mean_
std_devs_ = np.sqrt(scaler_part.var_)

orig_coefs = std_coefs_ / std_devs_
orig_intercept = std_intercept_ - np.sum((means_ * std_coefs_) / std_devs_)

print(f"Original Coefficients are: {orig_coefs}")
print(f"Original Intercept is: {orig_intercept}")

# Prediction equation (using original X)
print(f"The y_pred equation is as follows: {X @ orig_coefs + orig_intercept}")

# Model Visualization (using standardized X_train and extracted coefficients)
X_train_std = scaler_part.transform(X_train)

plt.figure()
plt.scatter(X_train_std[:, 0], y_train, color="blue")
plt.plot(X_train_std[:, 0], std_coefs_[0] * X_train_std[:, 0] + std_intercept_, "-r")
plt.xlabel("Standardized Engine size")
plt.ylabel("CO2 Emissions")
plt.show()

# Example with new data (using the pipeline for scaling)
new_data = np.array([[2, 33]])  # Example new data
new_predictions = pipeline.predict(new_data)  # Pipeline handles the scaling!
print(f"Predicted CO2 Emissions (with Pipeline): {new_predictions}")
