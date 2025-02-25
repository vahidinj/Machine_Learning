import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline

# OBJECTIVE: Classification with logistic regression to determine who is more likely to leave a company.

# Data Loading
df_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
df = pd.read_csv(df_url)[
    ["tenure", "age", "address", "income", "ed", "employ", "equip", "churn"]
]

df.sample(5)

# Data Preprocessing/Familiarization
df["churn"] = df["churn"].astype(int)
df.sample(5)
df.info()
df.describe()


# Data Visualization
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True)
plt.figure()
sns.histplot(data=df, x="churn")
sns.pairplot(data=df)
plt.show()

# NOTE: The histplot shows an imbalance in the data. Logistic Regression can be sensitive to data imbalance.

# MODEL BUILDING
X = np.asarray(df.drop(["churn"], axis=1))
y = np.asarray(df["churn"])

# Splitting the data: test, train, split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Pipeline
pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LogisticRegression())])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(y_pred)

# The models prediction/probabilities per instance
y_pred_proba = pipeline.predict_proba(X_test)
print(y_pred_proba)

# MODEL EVALUATION
loss = log_loss(y_test, y_pred_proba)
print(f"Log Loss: {loss}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision, Recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# The number of predictions the model got right and wrong and the accuracy of the model on the test set, which is the proportion of correctly classified instances
num_right = (y_pred == y_test).sum()
num_wrong = (y_pred != y_test).sum()
print(f"The number of correct predictions: {num_right}")
print(f"The number of wrong predictions: {num_wrong}")
print(f"The model accuracy: {num_right / len(y_test)}")

# Model Visualization
# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# The ROC graph indicates that the best value for threshold is somewhere between 0.3 and 0.4

# This (above) is with the threshold classification set to 0.5, meaning anything above 50% is set to 1 and below to 0.
# To adjust the threshold see below:
threshold = 0.35
# Classify based on the adjusted threshold
# The line below determines if the probability is greater or equal to threshold and if true it returns a value of 1, in turn increasing the number of values that return 1.
y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)
print(y_pred_adjusted)

# Function to evaluate model performance at different thresholds
thresholds = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]


def evaluate_threshold(threshold):
    y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)
    # Convert the 0/1 predictions back into a 2D probability distribution
    y_pred_adjusted_proba = np.column_stack((1 - y_pred_adjusted, y_pred_adjusted))

    loss_adjusted = log_loss(y_test, y_pred_adjusted_proba)
    cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)
    precision_adjusted = precision_score(y_test, y_pred_adjusted)
    recall_adjusted = recall_score(y_test, y_pred_adjusted)
    f1_adjusted = f1_score(y_test, y_pred_adjusted)
    accuracy_adjusted = (y_pred_adjusted == y_test).mean()

    print(f"Threshold: {threshold}")
    print(f"log_loss: {loss_adjusted}")
    print("Confusion Matrix (Adjusted):\n", cm_adjusted)
    print(f"Precision (Adjusted): {precision_adjusted}")
    print(f"Recall (Adjusted): {recall_adjusted}")
    print(f"F1-score (Adjusted): {f1_adjusted}")
    print(f"Accuracy (Adjusted): {accuracy_adjusted}")


# Evaluate model performance at different thresholds
for threshold in thresholds:
    evaluate_threshold(threshold)

# 'CONCLUSION' F1 score is indicative that the best threshold for the model is 0.35, as per the observations in the ROC Curve Graph, striking a good balance between precision and recall.
