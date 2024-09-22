# Customer Segmentation Project

## Overview
This project focuses on customer segmentation using unsupervised learning techniques. The goal is to identify distinct customer groups based on their transactional behavior and demographic features to enhance marketing strategies and improve customer satisfaction.

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Project Approach](#project-approach)
3. [Model Development](#model-development)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Key Findings](#key-findings)
6. [Visualizations](#visualizations)
7. [Conclusion](#conclusion)

## Dataset Description
The dataset used in this project includes several sheets:
- **Customers**: Information about customers, including IDs, join dates, cities, and genders.
- **Genders**: Mapping of gender IDs to gender names.
- **Cities**: Mapping of city IDs to city names.
- **Transactions**: Details of customer transactions, including IDs, transaction dates, statuses, and coupon usage.

## Project Approach
1. **Data Preprocessing**: 
   - Merged customer demographic data with transactional data.
   - Aggregated transaction data to summarize customer behavior (total transactions and subscribed count).
   - Encoded categorical features (gender and city) using one-hot encoding.

2. **Feature Selection**: 
   - Selected relevant features for segmentation, including demographic and transactional characteristics.

## Model Development
- **Clustering Algorithm**: K-Means clustering was used to segment customers.
- **Hyperparameter Tuning**: Explored different numbers of clusters (K) using the Elbow Method and Silhouette Score for optimization.

### Implementation Steps:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
customers_df = pd.read_excel('E-commerce_data.xlsx', sheet_name='customers')
genders_df = pd.read_excel('E-commerce_data.xlsx', sheet_name='genders')
cities_df = pd.read_excel('E-commerce_data.xlsx', sheet_name='cities')
transactions_df = pd.read_excel('E-commerce_data.xlsx', sheet_name='transactions')

# Merge data
df = customers_df.merge(genders_df, on='gender_id').merge(cities_df, on='city_id')
transaction_summary = transactions_df.groupby('customer_id').agg({
    'transaction_id': 'count',
    'transaction_status': lambda x: (x == 'subscribed').sum(),
}).rename(columns={'transaction_id': 'total_transactions', 'transaction_status': 'subscribed_count'}).reset_index()

df = df.merge(transaction_summary, on='customer_id', how='left').fillna(0)

# Feature selection
features = df[['gender_name', 'city_name', 'total_transactions', 'subscribed_count']]
features = pd.get_dummies(features, drop_first=True)

# Data scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-Means clustering
inertia = []
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

# Optimal clusters
optimal_k = 2 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)
```
## Evaluation Metrics
Inertia: Measures the compactness of the clusters; lower values indicate better clustering.
Silhouette Score: Evaluates the separation distance between clusters. Values closer to 1 indicate well-separated clusters.
## Key Findings
Identified 4 distinct customer segments with varying transaction behaviors.
Each segment's average transactions and subscription rates were analyzed.
Recommendations for coupon offerings were tailored based on segment characteristics to maximize customer loyalty and satisfaction.
## Visualizations
Count plots displaying customer distribution by gender and city.
Elbow Method and Silhouette Score plots to determine the optimal number of clusters.
``` python
Copy code
# Visualizing customer distribution by gender
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='gender_name')
plt.title('Customer Distribution by Gender')
plt.show()

# Visualizing customer distribution by city
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='city_name')
plt.title('Customer Distribution by City')
plt.xticks(rotation=45)
plt.show()

# Elbow Method and Silhouette Scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Distribution of customers across identified clusters
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='cluster')
plt.title('Customer Distribution by Cluster')
plt.show()
```
## Conclusion
This project successfully segmented customers using K-Means clustering, providing insights into distinct customer groups. The findings can be leveraged to enhance marketing strategies, focusing on targeted coupon offers to improve customer satisfaction and loyalty.
