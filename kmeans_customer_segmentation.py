"""
Customer Segmentation using K-Means Clustering
ML Mini Project - Unit 3 (Mixture Models and EM)

Groups 200 mall customers by Annual Income and Spending Score.
Run:  pip install pandas scikit-learn matplotlib
      python kmeans_customer_segmentation.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv("Mall_Customers.csv")
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = StandardScaler().fit_transform(df[features])

# 2. Elbow method (WCSS / inertia for K = 1..10)
ks = range(1, 11)
wcss = [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]
plt.plot(ks, wcss, "o-")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow Method")
plt.savefig("elbow.png", dpi=150)
plt.show()

# 3. Silhouette score (K = 2..10)
print("K  Silhouette")
for k in range(2, 11):
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    print(f"{k:<2} {silhouette_score(X, labels):.3f}")

# 4. Final model with K = 5
km = KMeans(5, n_init=10, random_state=0)
df["Cluster"] = km.fit_predict(X)
print("\nFinal silhouette score (K=5):", round(silhouette_score(X, df["Cluster"]), 3))

# 5. Plot clusters and centroids
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=df["Cluster"], cmap="viridis", s=40)
plt.scatter(*km.cluster_centers_.T, c="red", marker="X", s=250, label="Centroids")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segments (K = 5)")
plt.legend()
plt.savefig("clusters.png", dpi=150)
plt.show()

# 6. Segment profiles
print("\nSegment profiles (original units):")
print(df.groupby("Cluster")[["Age"] + features].mean().round(1))
print(df["Cluster"].value_counts().sort_index())
