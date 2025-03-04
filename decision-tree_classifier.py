import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

# OBJECTIVE: Use a decision tree classifier to determine which drug to choose based on features

# Data Loading and Familiarization
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(file_path)

# Display basic information and statistics about the dataset
df.info()
df.describe()

# Visualize the distribution of the 'Drug' categories
sns.countplot(data=df, x="Drug", order=sorted(df["Drug"].unique()))
plt.title("Drug Category Count")
plt.show()

# Custom mapping for 'Drug' column to numerical values
custom_map = {"drugA": 0, "drugB": 1, "drugC": 2, "drugX": 3, "drugY": 4}
df["Drug_num"] = df["Drug"].map(custom_map)

# Separate categorical and numerical features
df_cat = df.select_dtypes("object").drop(columns=["Drug"])
df_num = df.select_dtypes("number").drop(columns=["Drug_num"])

# Define a preprocessor to standardize numerical features and one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), df_num.columns),
        ("cat", OneHotEncoder(), df_cat.columns),
    ]
)

# Apply the preprocessor to the data
processed_data = preprocessor.fit_transform(df)

# Convert the processed data back to a DataFrame for better visualization
processed_df = pd.DataFrame(
    processed_data,
    columns=list(df_num.columns)
    + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(df_cat.columns)
    ),
)

# Visualize the correlation matrix of the processed data
sns.heatmap(processed_df.corr(numeric_only=True), cmap="coolwarm")
plt.show()

# Define a pipeline with the preprocessor and a decision tree classifier
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(criterion="entropy", max_depth=4)),
    ]
)

# Split the data into training and testing sets
X = df.drop(columns=["Drug", "Drug_num"])
y = df["Drug"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Fit the pipeline to the training data
pipeline.fit(X=X_train, y=y_train)

# Predict the test set results
tree_predict = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, tree_predict)
conf_matrix = confusion_matrix(y_test, tree_predict)
class_report = classification_report(y_test, tree_predict)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualize the decision tree
plt.figure(figsize=(12, 14))
plot_tree(
    pipeline.named_steps["classifier"],
    feature_names=processed_df.columns,
    class_names=list(custom_map.keys()),  # Convert dict_keys to list
    filled=True,
)
plt.show()
