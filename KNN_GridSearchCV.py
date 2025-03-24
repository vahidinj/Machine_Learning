import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# Load the iris dataset
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

# Define the pipeline with scaling, PCA, and KNN classifier
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),  # Standardize the data
        ("pca", PCA(n_components=2)),  # Reduce dimensions to 2
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=5),
        ),  # KNN classifier with 5 neighbors
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test data
test_score = pipeline.score(X_test, y_test)
print(f"Test Score: {test_score}")

# Predict the test data
y_pred = pipeline.predict(X_test)


# Function to plot the confusion matrix
def conf_matrix_plt(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
    plt.figure()
    sns.heatmap(
        data=conf_matrix,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Classification Confusion Matrix Pipeline")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# Plot the confusion matrix for the initial model
conf_matrix_plt(y_pred=y_pred, y_test=y_test)

# Hyperparameter Tuning

# Define the pipeline with scaling, PCA, and KNN classifier (without fixed parameters)
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),  # Standardize the data
        ("pca", PCA()),  # PCA without fixed number of components
        (
            "classifier",
            KNeighborsClassifier(),
        ),  # KNN classifier without fixed neighbors
    ]
)

# Define the parameter grid for GridSearchCV
parameter_search = {
    "pca__n_components": [2, 3, 4],  # Different numbers of PCA components to try
    "classifier__n_neighbors": [3, 4, 5, 6, 7],  # Different numbers of neighbors to try
}

# Define the cross-validation strategy
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV to find the best model
best_model = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_search,
    cv=cross_validation,
    scoring="accuracy",
    verbose=2,
)

# Fit the best model on the training data
best_model.fit(X_train, y_train)
best_model.best_params_

# Evaluate the best model on the test data
test_score = best_model.score(X_test, y_test)
print(f"Test Score from Best Model: {test_score}")

# Predict the test data using the best model
y_pred_bm = best_model.predict(X_test)

# Plot the confusion matrix for the best model
conf_matrix_plt(y_test=y_test, y_pred=y_pred_bm)
