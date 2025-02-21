import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


# OBJECTIVE: Simple linear regression to determine CO2 emissions based on vehicle features

# Data Loading

data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(data_url)


# Data Familiarization/Visualization

df.sample(5)
df.describe()
df_corr = df.corr(numeric_only=True)
df_corr

df = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
df.sample(5)

plt.figure()
sns.heatmap(df_corr, cmap="Reds")
df.hist()


fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Create a figure with 3 subplots

sns.scatterplot(data=df, x="ENGINESIZE", y="CO2EMISSIONS", hue="CYLINDERS", ax=axs[0])
axs[0].set_title("Engine Size vs CO2 Emissions")

sns.scatterplot(data=df, x="FUELCONSUMPTION_COMB", y="CO2EMISSIONS", ax=axs[1])
axs[1].set_title("Fuel Consumption vs CO2 Emissions")

sns.scatterplot(data=df, x="CYLINDERS", y="CO2EMISSIONS", ax=axs[2])
axs[2].set_title("Cylinders vs CO2 Emissions")

plt.tight_layout()

# plt.show()


# Because fuel consumption is self evidently a CO2 contributor ... enginesize will be used for model building
# Model Building

X = df["ENGINESIZE"].to_numpy()
y = df["CO2EMISSIONS"].to_numpy()

lin_reg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


lin_reg.fit(X_train.reshape(-1, 1), y_train)

print(f"Coefficients: {lin_reg.coef_}")
print(f"Intercept: {lin_reg.intercept_}")

# Model Evaluations

y_pred = lin_reg.predict(X_test.reshape(-1, 1))
print(f"Predicted_Values_Matrix: -> {y_pred}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_pred=y_pred, y_true=y_test)}")
print(f"Root MSE: {np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))}")
print(f"Root MSE: {root_mean_squared_error(y_pred=y_pred, y_true=y_test)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_pred=y_pred, y_true=y_test)}")
print(f"R2-Score: {r2_score(y_pred=y_pred, y_true=y_test)}")

# Model Visialization

plt.figure(figsize=(12, 10))
sns.scatterplot(x=X_test, y=y_test)
plt.plot(X_test, lin_reg.coef_ * X_test + lin_reg.intercept_, "red")

plt.show()

# Enable codeline bellow for comparison of built model vs sns.regplot

# plt.figure(figsize=(12, 10))
# sns.scatterplot(x=X_test, y=y_test, label="Test Data")
# plt.plot(
#     X_test,
#     lin_reg.coef_ * X_test + lin_reg.intercept_,
#     "red",
#     label="Sklearn Regression Line",
# )
# sns.regplot(
#     data=df,
#     x="ENGINESIZE",
#     y="CO2EMISSIONS",
#     scatter=False,
#     color="blue",
#     label="Seaborn Regression Line",
# )
# plt.legend()
# plt.show()
