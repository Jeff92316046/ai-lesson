import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
training_df = pd.read_csv("lesson8/california_housing_train.csv")
# 選擇 'median_income' 作為特徵
X = training_df[['median_income']]  # 使用雙括號 [[ ]] 選擇單個特徵，使其成為 DataFrame

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 K-means 分析，例如分成 3 個群集
kmeans = KMeans(n_clusters=2, random_state=0) 
kmeans.fit(X_scaled)

# 获取群集标签
labels = kmeans.labels_

# 将群集标签添加到原始 DataFrame
training_df['income_cluster'] = labels

# 顯示結果
print(training_df['income_cluster'])  # 顯示 'median_income' 和 'income_cluster' 欄位
temp = [0,0]
for i in training_df['income_cluster']:
    temp[i]+=1
print(temp)
print(kmeans.cluster_centers_)  # 顯示群集中心
training_df.to_csv("lesson8/output.csv",index=False)