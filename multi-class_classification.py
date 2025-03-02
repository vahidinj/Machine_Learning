import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report


# OBJECTIVE: Obesity risk prediction via implamentation of logistic regression for multi-class classification


# Data Loading/Familiarization

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
df = pd.read_csv(file_path)
df.head()
df.describe()
df.info()


sns.countplot(data=df, y="NObeyesdad")
plt.title("Obesity Levels")
plt.show()


# Feature scailing, one-hot encoding, OneVsOne(Rest)Classifier

df_cont = df.select_dtypes(include=["float"])
df_cat = df.select_dtypes(include="object").drop("NObeyesdad", axis=1)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), df_cont.columns),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), df_cat.columns),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", OneVsOneClassifier(LogisticRegression(max_iter=1000))),
        # ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ]
)

X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# Model evaluation and visualization


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X=X_test)

# Accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy: {np.round(accuracy * 100, 2)}")

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(
    data=cm,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
