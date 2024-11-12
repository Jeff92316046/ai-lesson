from artlib import ART1
import numpy as np

# Define input dataset
data = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 0]
])

# Set parameters
rho = 0.51   # Vigilance parameter
beta = 1.0   # Learning rate
L = 4.5      # Choice parameter

# Initialize the ART1 model
model = ART1(rho=rho, beta=beta, L=L)

# Fit model to data
model.fit(data)

# Predict cluster assignments
clusters = model.predict(data)

# Output each data point's cluster
for i, cluster in enumerate(clusters):
    print(f"Data sample {i + 1} belongs to cluster {cluster}")

# Optional: View cluster weight vectors
# print("\nCluster weights (centers):")
# for i, weight in enumerate(model.W):
#     print(f"Cluster {i + 1}: {weight}")
print(model.get_cluster_centers())
