# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
The aim is to predict daily temperature, PM2.5 pollution level, and Energy using environmental sensor data and the Random Forest Algorithm. The dataset used is **`weather-station-eee-block_2024_07_13.csv`**, which contains various environmental sensor readings.



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the weather sensor data.
2. Extract relevant environmental and time features.
3. Split the data into training and testing sets.
4. Train Random Forest models for temperature, PM2.5, and energy.
5. Predict the outputs and calculate MAE, RMSE, and R².

## Program:

Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.

Developed by: DHAN VARSHINI J P

RegisterNumber: 212224230055 

```
# ============================================================
# K-MEANS CLUSTERING ON WEATHER STATION DATA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "/content/weather-station-eee-block_2024_07_13.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
display(df.head())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nAvailable Columns:")
print(df.columns.tolist())


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
display(df.isnull().sum())


# ============================================================
# 4. CONVERT NUMERIC-LIKE COLUMNS TO NUMERIC
# ============================================================

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")


# ============================================================
# 5. SELECT NUMERIC FEATURES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumeric Columns:")
print(numeric_columns)


# ------------------------------------------------------------
# Remove columns that are likely IDs
# ------------------------------------------------------------

id_columns = []

for col in numeric_columns:

    col_lower = col.lower()

    if (
        "id" in col_lower
        or "index" in col_lower
        or "serial" in col_lower
        or "no." in col_lower
        or col_lower == "no"
    ):
        id_columns.append(col)

numeric_features = [
    col for col in numeric_columns
    if col not in id_columns
]

print("\nFeatures Available for Clustering:")
print(numeric_features)


# ============================================================
# 6. CHECK NUMBER OF FEATURES
# ============================================================

if len(numeric_features) < 2:
    raise ValueError(
        "Not enough numeric columns available for clustering. "
        "Please check the dataset."
    )


# ============================================================
# 7. SELECT FEATURES
# ============================================================

# Use all suitable numeric weather features
features = numeric_features

print("\nFeatures Used for Clustering:")
for feature in features:
    print("-", feature)


X = df[features].copy()


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
display(X.isnull().sum())

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values using median
X = X.fillna(X.median())

print("\nMissing values after cleaning:")
display(X.isnull().sum())


# ============================================================
# 9. STANDARDIZE DATA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully!")

print("Scaled Data Shape:", X_scaled.shape)


# ============================================================
# 10. ELBOW METHOD
# ============================================================

inertia = []

# Maximum possible clusters should be less than number of samples
max_k = min(10, len(X_scaled) - 1)

K_range = range(1, max_k + 1)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method for Optimal k")
plt.grid(True)

plt.show()


# ============================================================
# 11. SILHOUETTE METHOD
# ============================================================

sil_scores = []

silhouette_range = range(
    2,
    max_k + 1
)

for k in silhouette_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    sil_scores.append(score)


plt.figure(figsize=(8, 5))

plt.plot(
    silhouette_range,
    sil_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid(True)

plt.show()


# ============================================================
# 12. FIND BEST k USING SILHOUETTE SCORE
# ============================================================

best_k = list(silhouette_range)[
    np.argmax(sil_scores)
]

best_score = max(sil_scores)

print("\nBest k according to Silhouette Score:", best_k)
print("Best Silhouette Score:", round(best_score, 4))


# ============================================================
# 13. APPLY FINAL K-MEANS
# ============================================================

k_final = best_k

kmeans = KMeans(
    n_clusters=k_final,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels


# ============================================================
# 14. CLUSTER COUNTS
# ============================================================

print("\nCluster Counts:")

cluster_counts = df["Cluster"].value_counts().sort_index()

display(cluster_counts)


# ============================================================
# 15. CLUSTER CENTERS
# ============================================================

centers_scaled = kmeans.cluster_centers_

# Convert centers back to original units
centers_original = scaler.inverse_transform(
    centers_scaled
)

centers_df = pd.DataFrame(
    centers_original,
    columns=features
)

centers_df["Cluster"] = range(k_final)

print("\nCluster Centers in Original Units:")

display(
    centers_df.round(2)
)


# ============================================================
# 16. FINAL SILHOUETTE SCORE
# ============================================================

final_silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

print(
    "\nFinal Silhouette Score:",
    round(final_silhouette, 4)
)


# ============================================================
# 17. VISUALIZATION
# ============================================================

# For visualization, use the first two numeric features
plot_x = features[0]
plot_y = features[1]

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x=plot_x,
    y=plot_y,
    hue="Cluster",
    palette="tab10",
    s=80
)

# Plot cluster centers
plt.scatter(
    centers_df[plot_x],
    centers_df[plot_y],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.xlabel(plot_x)
plt.ylabel(plot_y)

plt.title(
    f"Weather Station Clustering using K-Means (k={k_final})"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 18. CLUSTER SUMMARY
# ============================================================

print("\nCluster Summary:")

cluster_summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

display(cluster_summary)


# ============================================================
# 19. SAVE CLUSTERED DATASET
# ============================================================

output_file = "/content/weather_station_clustered.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nClustered dataset saved successfully!")
print("File:", output_file)
```


## Output:

<img width="1368" height="741" alt="image" src="https://github.com/user-attachments/assets/c9fad554-6403-4e1f-a621-ba2bbe941dd4" />

<img width="750" height="555" alt="image" src="https://github.com/user-attachments/assets/23450137-43b3-46af-bb6a-561de364fcf2" />

<img width="562" height="600" alt="image" src="https://github.com/user-attachments/assets/70224380-4964-4fb5-aff0-5a6b6ea5a460" />

<img width="955" height="504" alt="image" src="https://github.com/user-attachments/assets/ff56faf7-ac92-4877-817c-e10c8fccae28" />

<img width="952" height="808" alt="image" src="https://github.com/user-attachments/assets/d98dd999-2348-472a-9cf1-508706120522" />




## Result:
Thus the program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm is Completed.
