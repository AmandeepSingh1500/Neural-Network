import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # Parameters for the triple pendulum dynamics
# g = 9.8
# L = 1.0

# # Define the ODE system for the analytical solution (using solve_ivp)
# def quadruple_pendulum_ode_scipy(t, y):
#     theta1, omega1, theta2, omega2, theta3, omega3, theta4, omega4 = y
#     eq1 = omega1
#     eq1_dot = -(g/(4*L)) * np.sin(theta1) \
#               - (3/4) * omega2 * np.cos(theta1 - theta2) \
#               - (3/4) * omega2**2 * np.sin(theta1 - theta2) \
#               - (1/2) * omega3 * np.cos(theta1 - theta3) \
#               - (1/2) * omega3**2 * np.sin(theta1 - theta3) \
#               - (1/4) * omega4 * np.cos(theta1 - theta4) \
#               - (1/4) * omega4**2 * np.sin(theta1 - theta4)

#     eq2 = omega2
#     eq2_dot = -(g/(3*L)) * np.sin(theta2) \
#               - omega1 * np.cos(theta1 - theta2) \
#               + omega1**2 * np.sin(theta1 - theta2) \
#               - (2/3) * omega3 * np.cos(theta2 - theta3) \
#               - (2/3) * omega3**2 * np.sin(theta2 - theta3) \
#               - (1/3) * omega4 * np.cos(theta2 - theta4) \
#               - (1/3) * omega4**2 * np.sin(theta2 - theta4)

#     eq3 = omega3
#     eq3_dot = -(g/(2*L)) * np.sin(theta3) \
#               - omega1 * np.cos(theta1 - theta3) \
#               + omega1**2 * np.sin(theta1 - theta3) \
#               - omega2 * np.cos(theta2 - theta3) \
#               + omega2**2 * np.sin(theta2 - theta3) \
#               - (1/2) * omega4 * np.cos(theta3 - theta4) \
#               - (1/2) * omega4**2 * np.sin(theta3 - theta4)

#     eq4 = omega4
#     eq4_dot = -(g/L) * np.sin(theta4) \
#               - omega1 * np.cos(theta1 - theta4) \
#               + omega1**2 * np.sin(theta1 - theta4) \
#               - omega2 * np.cos(theta2 - theta4) \
#               + omega2**2 * np.sin(theta2 - theta4) \
#               - omega3 * np.cos(theta3 - theta4) \
#               + omega3**2 * np.sin(theta3 - theta4)

#     return [eq1, eq1_dot, eq2, eq2_dot, eq3, eq3_dot, eq4, eq4_dot]


def quadruple_pendulum_ode_dimensionless(tau, y):
    theta1, omega1, theta2, omega2, theta3, omega3, theta4, omega4 = y

    # Pendulum 1
    eq1 = omega1
    eq1_dot = -(1/4) * np.sin(theta1) \
              - (3/4) * omega2 * np.cos(theta1 - theta2) \
              - (3/4) * omega2**2 * np.sin(theta1 - theta2) \
              - (1/2) * omega3 * np.cos(theta1 - theta3) \
              - (1/2) * omega3**2 * np.sin(theta1 - theta3) \
              - (1/4) * omega4 * np.cos(theta1 - theta4) \
              - (1/4) * omega4**2 * np.sin(theta1 - theta4)

    # Pendulum 2
    eq2 = omega2
    eq2_dot = -(1/3) * np.sin(theta2) \
              - omega1 * np.cos(theta1 - theta2) \
              + omega1**2 * np.sin(theta1 - theta2) \
              - (2/3) * omega3 * np.cos(theta2 - theta3) \
              - (2/3) * omega3**2 * np.sin(theta2 - theta3) \
              - (1/3) * omega4 * np.cos(theta2 - theta4) \
              - (1/3) * omega4**2 * np.sin(theta2 - theta4)

    # Pendulum 3
    eq3 = omega3
    eq3_dot = -(1/2) * np.sin(theta3) \
              - omega1 * np.cos(theta1 - theta3) \
              + omega1**2 * np.sin(theta1 - theta3) \
              - omega2 * np.cos(theta2 - theta3) \
              + omega2**2 * np.sin(theta2 - theta3) \
              - (1/2) * omega4 * np.cos(theta3 - theta4) \
              - (1/2) * omega4**2 * np.sin(theta3 - theta4)

    # Pendulum 4
    eq4 = omega4
    eq4_dot = -np.sin(theta4) \
              - omega1 * np.cos(theta1 - theta4) \
              + omega1**2 * np.sin(theta1 - theta4) \
              - omega2 * np.cos(theta2 - theta4) \
              + omega2**2 * np.sin(theta2 - theta4) \
              - omega3 * np.cos(theta3 - theta4) \
              + omega3**2 * np.sin(theta3 - theta4)

    return [eq1, eq1_dot, eq2, eq2_dot, eq3, eq3_dot, eq4, eq4_dot]

# Initial conditions for solve_ivp
y0 = [np.pi / 5.7, 0.0, np.pi / 15.9, 0.0, np.pi / 20, 0.0, np.pi / 30, 0.0]  # [theta1, omega1, theta2, omega2, theta3, omega3, theta4, omega4]
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve using solve_ivp
sol = solve_ivp(quadruple_pendulum_ode_dimensionless, t_span, y0, t_eval=t_eval, method='RK45')
theta1_analytical, theta2_analytical, theta3_analytical, theta4_analytical = sol.y[0], sol.y[2], sol.y[4], sol.y[6]

# theta1_analytical = np.degrees(theta1_analytical)
# theta2_analytical = np.degrees(theta2_analytical)
# theta3_analytical = np.degrees(theta3_analytical)
# theta4_analytical = np.degrees(theta4_analytical)

# Prepare target data for all three angles
y_data_all = np.stack([theta1_analytical, theta2_analytical, theta3_analytical,theta4_analytical], axis=1)
y_tensor_all = torch.tensor(y_data_all, dtype=torch.float32).to(device)  # Shape (N, 3)

# Convert input time data to PyTorch tensor
x_data = t_eval  # Use the same time values as in the numerical solution
x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1).to(device)  # Shape (N, 1)

# Define the Neural Network
class NeuralNetMultiOutput(nn.Module):
    def __init__(self):
        super(NeuralNetMultiOutput, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 4)  # Output layer: 3 outputs (theta1, theta2, theta3)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the network, loss function, and optimizer
model = NeuralNetMultiOutput().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop with Real-Time Loss Monitoring
epochs = 2000
losses = []

# plt.ion()  # Turn on interactive mode for real-time plotting
# fig, ax = plt.subplots()
# line, = ax.plot([], [], label='Loss')
# ax.set_xlim(0, epochs)
# ax.set_ylim(0, 0.1)  # Adjust this dynamically during training
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# ax.set_title('Training Loss')
# ax.legend()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(x_tensor)  # Forward pass
    loss = loss_fn(predictions, y_tensor_all)  # Compute loss (MSE over all three angles)
    
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    losses.append(loss.item())
    
    # # Update real-time plot
    # if epoch == 0:
    #     ax.set_ylim(0, loss.item())  # Dynamically adjust y-axis for the first epoch
    # line.set_xdata(range(len(losses)))
    # line.set_ydata(losses)
    # ax.relim()
    # ax.autoscale_view()
    # plt.pause(0.01)  # Pause for a moment to update the plot
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

plt.ioff()  # Turn off interactive mode
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_all = model(x_tensor).cpu().numpy()  # Predictions for all angles

# Plot the solutions and errors
plt.figure(figsize=(14, 6))

# Plot numerical and NN solutions for each angle
labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"]
colors = ['red', 'green', 'blue', 'yellow']

for i in range(4):
    plt.plot(x_data, y_data_all[:, i], label=f"{labels[i]} (Numerical)", color=colors[i], linestyle='dashed')
    plt.plot(x_data, y_pred_all[:, i], label=f"{labels[i]} (NN Approximation)", color=colors[i])

plt.xlabel('Time (t)')
plt.ylabel(r'$\theta$')
plt.title('Numerical and Neural Network Approximations of Quadrople Pendulum')
plt.legend()
plt.grid(True)
plt.show()

# Plot errors for each angle
plt.figure(figsize=(14, 6))
for i in range(4):
    error = np.abs(y_data_all[:, i] - y_pred_all[:, i])
    plt.plot(x_data, error, label=f"Error in {labels[i]}", color=colors[i])

plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.title('Numerical Error for Neural Network Approximations')
plt.legend()
plt.grid(True)
plt.show()
