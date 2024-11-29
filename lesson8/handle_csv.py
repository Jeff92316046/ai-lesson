import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
csv_file = pd.read_csv("lesson8/california_housing_train.csv")
csv_file["distance_with_zero"] = pow((csv_file["longitude"]**2 + csv_file["latitude"]**2),1/2)
pca = PCA(n_components=1)
csv_file['log_total_rooms'] = np.log(csv_file['total_rooms'])
csv_file['log_total_bedrooms'] = np.log(csv_file['total_bedrooms'])
csv_file['log_population'] = np.log(csv_file['population'])
csv_file['log_households'] = np.log(csv_file['households'])
pca.fit(csv_file.to_numpy(dtype=float))
print(csv_file.to_numpy(dtype=float))
# print(pca.transform(csv_file))
csv_file.to_csv("output.csv",index=False)  