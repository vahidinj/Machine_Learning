import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

# OBJECTIVE: Customer segmentation on historical data

# Load the dataset
df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv"
)

# Display basic information about the dataset
df.info()
df.describe()

# Drop unnecessary columns and handle missing values
df.drop(["Address", "Customer Id"], axis=1, inplace=True)
df.dropna(inplace=True)

# Extract feature values
X = df.values
cluster_num = 4

# Create a pipeline with StandardScaler and KMeans
pipline = Pipeline(
    [
        ("preprocessor", StandardScaler()),  # Standardize the features
        (
            "classifier",
            KMeans(init="k-means++", n_clusters=cluster_num, n_init=10),
        ),  # Apply KMeans clustering
    ]
)

# Fit the pipeline to the data
pipline.fit(X=X)

# Predict cluster labels
labels = pipline.predict(X)

# Add cluster labels to the DataFrame
df["Clust_km"] = labels

# Calculate cluster centroids
cluster_centroids = df.groupby("Clust_km").mean()

# Evaluate the model using inertia and silhouette score
inertia = pipline.named_steps["classifier"].inertia_
silhouette_avg = silhouette_score(X=X, labels=labels)

print(f"Inertia: {inertia}")
print(f"Silhouette Score: {silhouette_avg}")

# Elbow Method for model evaluation
inertia_values = []
cluster_range = range(1, 15)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow method results
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, inertia_values, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

# Visualization of clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x="Age",
    y="Income",
    size="Edu",
    hue="Clust_km",
    alpha=0.7,
    palette="dark",
    sizes=(70, 450),
)
plt.show()

# 3D scatter plot visualization
fig = px.scatter_3d(data_frame=df, x="Age", y="Income", z="Edu", color="Clust_km")
fig.show()
