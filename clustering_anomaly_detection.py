# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Step 1: Load the dataset
data = pd.read_csv('/home/signal/PycharmProjects/PythonProject/sample_data.csv')
print(data.head())

# Step 2: Data Cleaning
data = data.dropna()

# Step 3: Extract relevant features
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Label']
dataset = data[features]

# Step 4: Apply K-Means Clustering
X = dataset[['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']]
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X)
dataset['Cluster'] = clusters

# Visualize Clusters
plt.scatter(X['Flow Duration'], X['Total Fwd Packets'], c=clusters, cmap='viridis')
plt.xlabel('Flow Duration')
plt.ylabel('Total Fwd Packets')
plt.title('K-Means Clustering')
plt.show()

#Apply Isolation Forest for Anomaly Detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X)
dataset['Anomaly'] = anomalies
anomalous_data = dataset[dataset['Anomaly'] == -1]
print(anomalous_data.head())

# Heat Map for Network Traffic Features
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Heat Map of Network Traffic Features')
plt.show()

# Scatter Plot for Clustering Results
plt.scatter(X['Flow Duration'], X['Total Fwd Packets'], c=clusters, cmap='viridis')
plt.xlabel('Flow Duration')
plt.ylabel('Total Fwd Packets')
plt.title('Scatter Plot of Clustering Results')
plt.show()

# Bar Chart for Anomaly Distribution
anomaly_counts = dataset['Anomaly'].value_counts()
anomaly_counts.plot(kind='bar')
plt.title('Distribution of Anomalies')
plt.xlabel('Anomaly')
plt.ylabel('Count')
plt.show()
