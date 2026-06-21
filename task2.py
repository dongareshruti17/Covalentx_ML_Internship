import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load Dataset
cust_data = pd.read_csv("customer_shopping_data.csv")

# Select categorical columns
cust_encode = cust_data[['gender','category','payment_method','shopping_mall']].copy()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in cust_encode.columns:
    cust_encode[c] = le.fit_transform(cust_encode[c])

# Numerical columns
num_data = cust_data[['age', 'quantity', 'price']]
# Final Dataset
cust_dataset = pd.concat([num_data, cust_encode], axis=1)

print("\nDataset Used For Clustering:")
print(cust_dataset)
# K-Means Clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, random_state=42)
km.fit(cust_dataset)
labels = km.labels_
centroids = km.cluster_centers_
# Add cluster column
cust_dataset['Cluster'] = labels
# SSE Calculation
sse = []
for i in range(4):
    cluster_points = cust_dataset[cust_dataset['Cluster'] == i]
    cluster_data = cluster_points.drop('Cluster', axis=1)
    centroid = centroids[i]
    error = np.sum((cluster_data.values - centroid) ** 2)
    sse.append(error)
print("\nSSE per Cluster:")
print(sse)
# Visualization
plt.figure(figsize=(8,5))
plt.scatter(cust_dataset['quantity'],cust_dataset['price'],c=labels)
plt.scatter(centroids[:,1],centroids[:,2],marker='+',color='red',label='Centroids')
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.title("Customer Segmentation using K-Means Clustering")
plt.legend()
plt.show()

