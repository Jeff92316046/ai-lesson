import numpy as np
import matplotlib.pyplot as plt

# 定義輸入矩陣 (4x4)
inputs = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 定義輸出值
outputs = np.array([1, -1, -2, 2])

# 初始化參數
learning_rate = 0.9  # 學習率
iterations = 40    # 迭代次數
m = outputs.shape[0]  # 樣本數量

# 隨機初始化權重
w = np.random.randn(inputs.shape[1])

# 用來記錄每次迭代的損失值和權重變化
loss_history = []
weights_history = []

# 定義損失函數 (均方誤差)
def compute_loss(predictions, outputs):
    return np.mean((predictions - outputs) ** 2)

# 梯度下降法
for t in range(iterations):
    # 計算預測值
    predictions = np.dot(inputs, w)
    
    # 計算誤差
    error = predictions - outputs
    
    # 計算梯度 (均方誤差對權重的偏導數)
    gradient = (2/m) * np.dot(inputs.T, error)
    
    # 更新權重
    w = w - learning_rate * gradient
    
    # 每次迭代儲存損失值和權重
    loss = compute_loss(predictions, outputs)
    loss_history.append(loss)
    weights_history.append(w.copy())
    
    # 每 100 次迭代輸出一次損失值和權重
    print(f"Iteration {t}: Loss = {loss}, Weights = {w}")

# 輸出最終的權重
print("最終權重:", w)

# 繪製損失值的收斂圖
plt.figure(figsize=(12, 6))

# 繪製損失值收斂曲線
plt.subplot(1, 2, 1)
plt.plot(range(iterations), loss_history, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Loss Convergence")
plt.grid(True)

# 繪製權重變化曲線
plt.subplot(1, 2, 2)
weights_history = np.array(weights_history)
for i in range(weights_history.shape[1]):
    plt.plot(range(iterations), weights_history[:, i], label=f'w{i+1}')
plt.xlabel("Iteration")
plt.ylabel("Weight Values")
plt.title("Weights Convergence")
plt.legend()
plt.grid(True)

# 顯示圖形
plt.tight_layout()
plt.show()
