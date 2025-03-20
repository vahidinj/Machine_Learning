# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# OBJECTIVE: Build a classifier to predict service category (custcat).

# Load the dataset
df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
)

# Display basic information about the dataset
df.head()
df.info()
df.describe()
df["custcat"].value_counts()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data=df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()

# Calculate and sort the absolute correlation values with the target variable 'custcat'
corr_values = abs(df.corr()["custcat"].drop("custcat")).sort_values(ascending=False)

# Model Building

# Initialize the KNN model with k=4
k = 4
pipeline = Pipeline(
    [
        ("preprocessor", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ]
)

# Split the data into features (X) and target (y)
X = df.drop(
    ["custcat"],
    axis=1,
)
y = df["custcat"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Print the test accuracy score
print(f"Test Accuracy Score: {accuracy_score(y_test, y_pred=y_pred)}")

# Model Optimization

# Define a range of k values to test
k_range = range(1, 150)
k_scores = []

# Loop over the range of k values
for k in k_range:
    pipeline = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    # Perform cross-validation and store the mean accuracy score
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())

# Plot the accuracy scores for each k value
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker="o")
plt.xlabel("Value of k for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Finding the Best k for KNN")
plt.show()

# Find the best k value
best_k = k_range[np.argmax(k_scores)]
print(
    f"The best k value is {best_k} with a cross-validation accuracy of {max(k_scores):.2f}"
)
# Might want to adjust the train_test_split parameters, 91 neighbors seems high for a 1000 entry data set

# Train the final model with the best k value
pipeline = Pipeline(
    [
        ("preprocessor", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the final model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy Score with k={best_k}: {accuracy:.2f}")


# The model still does not perform all too well. Maybe removing some of the features that have a weak
# correlation with the target might imporve the model. Check corr_values.
# Sugested features to keep: ed, tenure, income, emply
