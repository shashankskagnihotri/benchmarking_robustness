import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Step 1: Read and preprocess error log files
file_names = []
error_logs = []

error_folder = "slurm/work_dir/log_error"

for file in os.listdir(error_folder):
    file_names.append(file)
    with open(os.path.join(error_folder, file), "r") as f:
        error_logs.append(f.read())

# Step 2: Convert text data into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(error_logs)

# Step 3: Apply hierarchical clustering
Z = linkage(X.toarray(), method="ward")

# Step 4: Visualize dendrogram to determine the number of clusters
plt.figure(figsize=(10, 5))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=20,
    leaf_rotation=90.0,
    leaf_font_size=12.0,
    show_contracted=True,
)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()

# Step 5: Cut the dendrogram to obtain clusters
max_d = 1  # Adjust this threshold based on the dendrogram
clusters = fcluster(Z, max_d, criterion="distance")

# Step 6: Assign file names to clusters
file_clusters = {}
for file, cluster in zip(file_names, clusters):
    file_clusters.setdefault(cluster, []).append(file)

# Step 7: Output file names in each cluster
for cluster, files in file_clusters.items():
    print(f"Cluster {cluster}:\n")
    for file in files:
        print(file)
    print("\n" * 2)
