import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Preprocess the data
# Drop irrelevant columns like CustomerID
data.drop("CustomerID", axis=1, inplace=True)

# Handle any missing values if present
data.dropna(inplace=True)

# Encode categorical variables
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Gender'])
    ],
    remainder='passthrough'
)
data_encoded = ct.fit_transform(data)
data_encoded = pd.DataFrame(data_encoded, columns=['Female', 'Male'] + data.columns[1:].tolist())

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_encoded)

# Perform customer segmentation using K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

# Evaluate the clustering
silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)

# Visualize the clusters
# For 2D visualization, consider plotting only two features (e.g., spending score vs. annual income)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()
