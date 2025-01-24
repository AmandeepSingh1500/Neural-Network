import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the system of equations
def lambda_nu_equations(R, y):
    alpha = 1
    nu, lam = y
    dnu_dR = (np.exp(lam))*(-alpha/(R**3) + 1/R) - 1/R  # Equation for nu'
    dlam_dR = (np.exp(lam))*(alpha/(R**3) - 1/R) + 1/R # Equation for lambda'
    return [dnu_dR, dlam_dR]

# Initial conditions for solve_ivp
y0 = [-2.5, 2.5]  # Initial conditions: nu(1) = -1, lambda(1) = 1
R_span = (1.0, 100.0)
R_eval = np.linspace(R_span[0], R_span[1], 1000)

# Solve using solve_ivp
sol = solve_ivp(lambda_nu_equations, R_span, y0, t_eval=R_eval, method='RK45')
nu_analytical, lambda_analytical = sol.y[0], sol.y[1]

# Prepare data for PyTorch
y_data_all = np.stack([nu_analytical, lambda_analytical], axis=1)
y_tensor_all = torch.tensor(y_data_all, dtype=torch.float32).to(device)
x_data = R_eval
x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1).to(device)

# Define the Neural Network
class NeuralNetMultiOutput(nn.Module):
    def __init__(self):
        super(NeuralNetMultiOutput, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 2)  # Output layer: 2 outputs (nu, lambda)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the network, loss function, and optimizer
model = NeuralNetMultiOutput().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 1000
losses = []
plt.ion()  # Turn on interactive mode for real-time plotting
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Loss')
ax.set_xlim(0, epochs)
ax.set_ylim(0, 0.1)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(x_tensor)  # Forward pass
    loss = loss_fn(predictions, y_tensor_all)  # Compute loss
    
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    losses.append(loss.item())
    
    if epoch == 0:
        ax.set_ylim(0, loss.item())  # Adjust y-axis dynamically
    line.set_xdata(range(len(losses)))
    line.set_ydata(losses)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

plt.ioff()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_all = model(x_tensor).cpu().numpy()

# Plot the solutions
plt.figure(figsize=(14, 6))
labels = [r"$\nu$", r"$\lambda$"]
colors = ['blue', 'orange']

for i in range(2):
    plt.plot(x_data, y_data_all[:, i], label=f"{labels[i]} (Numerical)", color=colors[i], linestyle='dashed')
    plt.plot(x_data, y_pred_all[:, i], label=f"{labels[i]} (NN Approximation)", color=colors[i])

plt.xlabel('R')
plt.ylabel('Values')
plt.title('Numerical and Neural Network Approximations')
plt.legend()
plt.grid(True)
plt.show()

# Plot errors
plt.figure(figsize=(14, 6))
for i in range(2):
    error = np.abs(y_data_all[:, i] - y_pred_all[:, i])
    plt.plot(x_data, error, label=f"Error in {labels[i]}", color=colors[i])

plt.xlabel('R')
plt.ylabel('Error')
plt.title('Numerical Error for Neural Network Approximations')
plt.legend()
plt.grid(True)
plt.show()
