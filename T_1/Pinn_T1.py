# Re-import necessary libraries as execution reset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import mpld3
import json

alpha_values_0 = [-0.05,-0.1,-0.5,0.05,0.1,0.5] #[0.05,0.1] + list(np.arange(0.5, 10, 0.5))  # [0.05,0.1,0.2,0.3,0.4,0.5,5,7]
#alpha_values_0 = [0.1]
#print(alpha_values_0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)


results_json = {}

for element in alpha_values_0:

    alpha_key = f"alpha_0_{element}"
    results_json[alpha_key] = {}
     

    # Neural network model for A(r)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 150),
                nn.Tanh(),
                nn.Linear(150, 150),
                nn.Tanh(),
                nn.Linear(150, 1),
                #nn.Softplus()  # Ensure A(r) > 0
            )

        def forward(self, r):
            return self.net(r)

    # Initialize model
    #model = Net()
    model = Net().to(device)

    # Example parameters (adjustable as per user's data)
    p = 2.0
    alpha_0 = element
    alpha_1 = 1.0

    # Define the PINN loss explicitly
    def pinn_loss(model, r,epoch):
        r.requires_grad = True
        A = model(r)
        lnA = torch.log(A)

        # First derivative calculations
        A_prime = torch.autograd.grad(A, r, grad_outputs=torch.ones_like(A), create_graph=True)[0]

        A_double_prime = torch.autograd.grad(A_prime, r, grad_outputs=torch.ones_like(A_prime), create_graph=True)[0]

        lnA_prime = A_prime / A

        Phi = r * lnA_prime
        #Phi_prime = torch.autograd.grad(Phi, r, grad_outputs=torch.ones_like(Phi), create_graph=True)[0]

        #Phi = (r*A_prime) / A        

        # Terms from the given simplified equation (with delta(r)=0)
        #term1 = (A**2)*Phi_prime*A_prime - (A_prime**2)*A + ((p + 1)/r*(p - 0.5))*(Phi_prime*(A**3)-A_prime*(A**2))




        term1 =  A_prime*A + r*A*A_double_prime - (A_prime**2)*r - (A_prime**2)*A  + ((p + 1)/r*(p - 0.5))*(  A_prime*(A**2) + r*(A_double_prime)*(A**2) - (A_prime**2)*A*r - (A**2)*A_prime  )
        
        term2 = -(3*r/2)*A_prime**3
        term3 = 3*(A_prime**2)*A -4*(A_prime*(A**2))/r
        term4 = (A_prime**2)*( (5*r*A_prime/2) - A)
        term5 = ((p - 1)/r*(p - 0.5))*A_prime*( (A**2) + A)
        term6 = ( 4/(r**2) )*( (A**2) - (A**3) )
        term7 = (alpha_0/(alpha_1*(p-0.5)) )*( (r*A_prime*A/2) - A**2 )

        
        # term1 = (Phi + (p + 1)/(p - 0.5)) * (Phi_prime/r - Phi/r**2)
        # term2 = lnA_prime**3
        # term3 = (3 * Phi - 4) * A_prime / (r * A)
        # term4 = (5 * Phi - 2) * A_prime**2 / (2 * A**2)
        # term5 = (A + 1)*(p - 1)*Phi / ((p - 0.5)*r**2 * A)
        # term6 = -4*(A - 1)/(r**2 * A)
        # term7 = alpha_0 * (Phi - 2) / (2 * alpha_1 * A * (p - 0.5))


        T1 = (term1 + term2 + term3 + term4 + term5 + term6 + term7)

        loss = torch.mean(T1**2)
        if epoch == 20000:
            print('T1:',T1)
            print('loss:',loss)    
        return loss

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_list = []
    # Training loop
    epochs = 20000
    for epoch in range(epochs+1):
        optimizer.zero_grad()
        #r = torch.linspace(0.1, 50, 5000).unsqueeze(1)  # Avoid r=0
        r = torch.linspace(0.1, 10, 10000, device=device).unsqueeze(1)

        loss = pinn_loss(model, r,epoch)

        # Initial conditions at r=0.1 explicitly enforced
        #r0 = torch.tensor([[0.1]], requires_grad=True)
        IC_r_value = 1.0
        r0 = torch.tensor([[IC_r_value]], requires_grad=True, device=device) # (A(1)=0, A'(1)=0)

        A0 = model(r0)                                                                                  # Value from model at A(0.1)  (expected 1)
        A0_prime = torch.autograd.grad(A0, r0, grad_outputs=torch.ones_like(A0), create_graph=True)[0]  # Value from model at A'(0.1) (expected 0)
        #A0_double_prime = torch.autograd.grad(A0_prime, r0, grad_outputs=torch.ones_like(A0_prime), create_graph=True)[0]

        IC_loss = (10**4)*(A0 - 0)**2 + 1*(A0_prime - 1)**2 # + (A0_double_prime - 0)**2  # A(0.1)=1, A'(0.1)=0 explicitly enforced (10**8)*

        loss = loss + 1*IC_loss

        #pdb.set_trace()
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()

        if float(loss.item()) < 10**(-3): # capping the training at this loss
            print( float(loss.item()) )
            break

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

            # For computing derivatives, r must require gradients
            r_test = torch.linspace(0.1, 10, 100000, device=device).unsqueeze(1)
            r_test.requires_grad = True
            A_pred = model(r_test)
            A_prime = torch.autograd.grad(
                A_pred, r_test,
                grad_outputs=torch.ones_like(A_pred),
                create_graph=False
            )[0]

            # Append the current epoch and the corresponding results to the dictionary.
            # Note: These lists are very long; you might consider saving only a subset.


            results_json[alpha_key][epoch] = {"A":0 , "A_prime":0 } 
            results_json[alpha_key][epoch]["A"] = [ float(i[0]) for i in A_pred.detach().cpu().tolist()] 
            results_json[alpha_key][epoch]["A_prime"] = [ float(i[0]) for i in A_prime.detach().cpu().tolist()] 

    # # Predict and plot
    # while True:




    # Define the base folder and subfolders for saving plots
    base_folder = r'D:\4th sem project\T1_IC_Begening(20K epochs a=1 and a_prime = 1 at r=1 with weight to IC_loss=1)'
    plot_folder = os.path.join(base_folder, 'Plots_alpha_0')
    interactive_plot_folder = os.path.join(base_folder, 'Interactive_plots_alpha_0')
    loss_folder = os.path.join(base_folder, 'Loss_alpha_0')
    IC_Value_folder = os.path.join(base_folder, 'IC_Value_folder')
    Models = os.path.join(base_folder, 'Models')

    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(loss_folder, exist_ok=True)
    os.makedirs(interactive_plot_folder, exist_ok=True)
    os.makedirs(IC_Value_folder, exist_ok=True)
    os.makedirs(Models, exist_ok=True)

    # Generate prediction plot
    r_test = torch.linspace(0.1, 10, 100000, device=device).unsqueeze(1)
    # To compute the derivative of A(r), r_test must require gradients
    r_test.requires_grad = True
    A_pred = model(r_test)
    
    # Compute the derivative A'(r)
    A_prime = torch.autograd.grad(
        A_pred, r_test,
        grad_outputs=torch.ones_like(A_pred),
        create_graph=False
    )[0]

    # Convert tensors to NumPy arrays and lists
    r_list = r_test.detach().cpu().numpy().flatten().tolist()
    A_list = A_pred.detach().cpu().numpy().flatten().tolist()
    A_prime_list = A_prime.detach().cpu().numpy().flatten().tolist()

    # For plotting, obtain predictions without gradient tracking
    A_pred_np = model(r_test.to(device)).detach().cpu().numpy()

    # Plot the PINN solution for A(r)
    plt.figure(figsize=(10, 6))
    plt.plot(r_test.detach().cpu().numpy(), A_pred_np, label='PINN Solution for A(r)', linewidth=2)
    plt.xlabel('r')
    plt.ylabel('A(r)')
    plt.title(f'PINN Solution of the Given Differential Equation (δ(r)=0), (alpha_0={alpha_0})')
    plt.grid(True)
    plt.legend()

    # Save the prediction plot as a PNG image
    plot_filename = os.path.join(plot_folder, f'pinn_solution_alpha0_{alpha_0}.png')
    plt.savefig(plot_filename)
    # plt.show()

    # Generate loss curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, linestyle='-', color='b', label='Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f'Loss Curve (Logarithmic Scale), (alpha_0={alpha_0})')
    plt.legend()

    # Save the loss curve plot as a PNG image
    loss_filename = os.path.join(loss_folder, f'loss_curve_alpha0_{alpha_0}.png')
    plt.savefig(loss_filename)
    # plt.show()

    # Create the interactive plot using mpld3
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_test.detach().cpu().numpy(), A_pred_np, label='PINN Solution for A(r)', linewidth=2)
    ax.set_xlabel('r')
    ax.set_ylabel('A(r)')
    ax.set_title(f'PINN Solution of the Given Differential Equation (δ(r)=0), (alpha_0={alpha_0})')
    ax.grid(True)
    ax.legend()

    # Save the interactive HTML file in the designated folder
    html_filename = os.path.join(interactive_plot_folder, f'pinn_solution_alpha0_{alpha_0}.html')
    mpld3.save_html(fig, html_filename)
    print(f"Interactive HTML plot saved at: {html_filename}")

    print('last value of A:', A_list[-1])
    print('last value of A_prime:', A_prime_list[-1])


    # Save the model's state dictionary to a file named 'model.pth'
    #prediction_filename = os.path.join(Models, f'IC_alpha0_{alpha_0}.txt')
    
    torch.save(model.state_dict(), os.path.join(Models,f"model_alpha0_({alpha_0}).pth") )

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # You can also save additional info such as the epoch number or loss
        'epoch': epoch,
        'loss': loss.item()
    }
    torch.save(checkpoint, os.path.join(Models, f"checkpoint_alpha0_({alpha_0}).pth") )


    txt_filename = os.path.join(IC_Value_folder, f'IC_alpha0_{alpha_0}.txt')
    
    

    pdb.set_trace()
    text1 = f"IC_A(r={IC_r_value}) = {A_list[ [ round(float(i[0]),4) for i in r_test].index(IC_r_value) ]}\n" 
    text2 = f"IC_A(r={IC_r_value}) = {A_prime_list [ [ round(float(i[0]),4) for i in r_test].index(IC_r_value) ] }\n"
    text3 = f"Loss = {float(loss.item())}"

    with open(txt_filename, "w") as f:
        f.write(text1)
        f.write(text2)
        f.write(text3)

    # After processing all α₀ values, write results_json to a JSON file.
    with open(f"predictions_({alpha_0}).json", "w") as json_file:
        json.dump(results_json, json_file)



import os
os.system("shutdown /s /t 0")