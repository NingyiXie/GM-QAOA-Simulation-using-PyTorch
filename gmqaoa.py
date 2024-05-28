import torch
import numpy as np
from itertools import combinations
import sys

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     # print("cuda is available")
# else:
#     device = torch.device("cpu")
#     # print("use cpus")
# sys.stdout.flush()
device = torch.device("cpu")

def generate_combinations_with_zero(p):
    # Initialize an empty list to store all possible combinations
    all_combinations = [np.array([0])]
    
    # Generate combinations for each length from 1 to p
    for i in range(1, p + 1):
        for combo in combinations(range(1, p + 1), i):
            # Convert the combination tuple to a list and insert 0 at the beginning
            
            combo_with_zero = [0] + list(combo)
            # Ensure the list is sorted (this step is actually not necessary with combinations)
            # combo_with_zero.sort()
            
            # Add the prepared list to the collection of all combinations
            all_combinations.append(np.array(combo_with_zero))
    return all_combinations


class GMQAOA:
    def __init__(self, rounds, distribution, opt_type = 'max' ,target_metric = 'p_opt', initial_gammas = [], initial_betas = []):
        
        self.rounds = rounds
        
        self.distribution = distribution
        
        self.size = torch.sum(self.distribution[:, 1])
        
        self.opt_type = opt_type
        if self.opt_type == 'max':
            self.opt_value = torch.max(self.distribution[:,0])
            self.opt_count = self.distribution[torch.argmax(self.distribution[:,0]),1]
        else:
            self.opt_value = torch.min(self.distribution[:,0])
            self.opt_count = self.distribution[torch.argmin(self.distribution[:,0]),1]
        
        self.combs = generate_combinations_with_zero(self.rounds)
        
        self.p_opt_value = 0
        self.approx_value = 0
        self.opt_gap_value = -1
        
        self.target_metric = target_metric
        if self.target_metric == 'p_opt':
            self.optimize_function = self.p_opt_coef
        else:
            self.optimize_function = self.expected_value_func
        
        if initial_gammas == [] or initial_betas == []:
            self.gammas = torch.randn(self.rounds, device=device, requires_grad=True)
            self.betas = torch.randn(self.rounds, device=device, requires_grad=True)
        else:
            self.gammas = torch.tensor(initial_gammas, device=device, requires_grad=True)
            self.betas = torch.tensor(initial_betas, device=device, requires_grad=True)
            
    def approx_func_of_obj(self,gammas):
        # Step 1: Prepare G for cumulative sum in reverse
        gammas_reversed_cumsum = torch.flip(torch.cumsum(torch.flip(gammas, dims=[0]), dim=0), dims=[0])
        # Step 2: Expand G and O to match the desired output shape
        gammas_expanded = gammas_reversed_cumsum.unsqueeze(1).expand(-1, self.distribution[:,0].shape[0])  # Shape: (p, N)
        obj_expanded = self.distribution[:,0].repeat(self.rounds, 1)  # Also (p, N), but this step might be redundant due to broadcasting rules
        # Step 3: Calculate the outer operation
        product = gammas_expanded * obj_expanded  # Element-wise multiplication suitable for broadcasting
        # Step 4: Apply the complex exponential function
        func_exp = torch.exp(-1j * product)
        # Adding the extra row for B[p, :], which should be the exponential of 0, resulting in ones
        func_of_obj = torch.cat([func_exp, torch.ones(1, self.distribution[:,0].shape[0], dtype=torch.cfloat, device = device)], dim=0)  # torch.cfloat to ensure complex type
        return func_of_obj
    
    def p_opt_func_of_obj(self,gammas):
        # Step 1: Prepare G for cumulative sum in reverse
        gammas_reversed_cumsum = torch.flip(torch.cumsum(torch.flip(gammas, dims=[0]), dim=0), dims=[0])
        # Step 2: Expand G and O to match the desired output shape
        gammas_expanded = gammas_reversed_cumsum.unsqueeze(1).expand(-1, 1)  # Shape: (p, 1)
        obj_expanded = torch.ones((self.rounds,1), device = device) * self.opt_value
        # Step 3: Calculate the outer operation
        product = gammas_expanded * obj_expanded
        # Step 4: Apply the complex exponential function
        func_exp = torch.exp(-1j * product)
        # Adding the extra row for B[p, :], which should be the exponential of 0, resulting in ones
        func_of_obj = torch.cat([func_exp, torch.ones(1, 1, dtype=torch.cfloat, device = device)], dim=0)  # torch.cfloat to ensure complex type
        return func_of_obj
        
    def inner_function(self,gammas,betas,func_of_obj):
        func_of_dist = torch.zeros((1,self.rounds+1), dtype=torch.cfloat, device = device)

        for comb in self.combs:
            sub_func = (-1)**(len(comb)-1)
            sub_func *= torch.prod(1 - torch.exp(-torch.tensor(1j) * betas[comb[1:]-1]))
            for i in range(1,len(comb)):
                sub_func *= torch.sum(self.distribution[:,1] * torch.exp(-torch.tensor(1j) * torch.sum(gammas[range(comb[i-1],comb[i])]) * self.distribution[:,0]))/self.size
            func_of_dist[0][comb[-1]] += sub_func
                
        return (func_of_dist @ func_of_obj)[0]
    
    def expected_value_func(self,gammas,betas): 
        func_of_obj = self.approx_func_of_obj(gammas)
        exp = torch.sum(self.distribution[:,0] * (self.distribution[:,1] * (self.inner_function(gammas,betas,func_of_obj).abs().pow(2)/self.size)))
        if self.opt_type == 'min':
            return exp
        else:
            return -exp
        
    def approx_ratio(self,exp):
        return exp/self.opt_value
    
    def p_opt_coef(self,gammas,betas):
        func_of_obj = self.p_opt_func_of_obj(gammas)
        return -self.inner_function(gammas,betas,func_of_obj)[0].abs().pow(2)
    
    def p_opt(self,coef):
        return coef*self.opt_count/self.size
        
    def initializing(self,lr=2e-2, max_step = 200):
        if self.rounds != 1:
            T = torch.tensor([1.], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([T], lr)
            for step in range(max_step):
                optimizer.zero_grad()
                loss = self.optimize_function(torch.arange(1, self.rounds + 1, device=device).float() * (T / self.rounds) / self.rounds, (1 - torch.arange(1, self.rounds + 1, device=device).float()/self.rounds) * (T / self.rounds))
                # print(f"Level {self.rounds}, Step {step}: loss={loss.item()}")
                loss.backward()
                optimizer.step()
            self.gammas = torch.arange(1, self.rounds + 1, device=device).float() * (T.item() / self.rounds) / self.rounds
            self.betas = (1 - torch.arange(1, self.rounds + 1, device=device).float()/self.rounds) * (T.item() / self.rounds)
            self.gammas.requires_grad_(True)
            self.betas.requires_grad_(True)
        
    def run(self,lr=4e-3,max_step = 1200):
        # his = []
        optimizer = torch.optim.Adam([self.gammas,self.betas], lr)        
        for step in range(max_step):
            optimizer.zero_grad()
            loss = self.optimize_function(self.gammas,self.betas)
            if step % (max_step//20) == 0:
                print(f"Level {self.rounds}, {self.target_metric}, Step {step}, loss={loss.item()}")
            loss.backward()
            optimizer.step()
            
        self.p_opt_value = self.p_opt(-self.p_opt_coef(self.gammas,self.betas)).item()
        self.approx_value = np.abs(self.approx_ratio(self.expected_value_func(self.gammas,self.betas)).item())
        self.opt_gap_value = np.abs(self.approx_ratio(self.expected_value_func(self.gammas,self.betas)).item())-1
            