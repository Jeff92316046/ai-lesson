#36 input and 1 output
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the perceptron neural network class
class PerceptronNetwork(nn.Module):
    def __init__(self):
        super(PerceptronNetwork, self).__init__()
        # Define the input layer (36 neurons)
        self.input_layer = nn.Linear(36, 10)
        # Define the output layer (1 neurons for 10 classes)
        self.output_layer = nn.Linear(10, 1)
        #output 10 neurons for 10 classes

    def forward(self, x):
        # Pass data through input_layer
        x = torch.sigmoid(self.input_layer(x))
        # Pass data through output_layer
        x = self.output_layer(x)
        return x

# Create an instance of the network
network = PerceptronNetwork()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)
# Define the input data (7-segment display data)
input_data = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num1
    [1, 0, 0, 0, 1, 1, 1, 1, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 1, 1, 1, 1, 0, 0, 0, 1],  # num2
    [1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num3
    [1, 1, 1, 1, 1, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num4
    [1, 1, 1, 1, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 1, 1, 1, 1],  # num5
    [1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 1, 1, 1, 1],  # num6
    [1, 0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num7
    [1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num8
    [1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  1, 1, 1, 1, 1, 0, 0, 0, 1],  # num9
    [1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 1,  1, 0, 0, 0, 0, 0, 0, 0, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num0
])
# Define the output data (corresponding to the input data)
output_data = np.array([
[1],
[2],
[3],
[4],
[5],
[6],
[7],
[8],
[9],
[0]
])
# Convert input and output data to torch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)

# Training loop
num_epochs = 10000
loss_values = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = network(input_tensor)
    loss = criterion(outputs, output_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
# Plot the learning curve
import matplotlib.pyplot as plt
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')

test = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0, 1,  1, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0, 0, 1, 1, 1, 1, 1],  # num5
    [1, 1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0, 0, 1, 0, 0, 0, 1,  0, 0, 0, 0, 1, 1, 1, 1, 1],  # num6
    [1, 1, 1, 0, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1],  # num7
])
# Display the complete output values after training
test_tensor = torch.tensor(test, dtype=torch.float32)
final_outputs = network(test_tensor)
print("Final Output Values:")
print(final_outputs)
#36 input 10 output
plt.show()
