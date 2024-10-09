import torch
import numpy as np
import time


def clamp_threshold(C,th):
    min_C = C[:, 0].min()
    max_C = C[:, 0].max()
    return torch.clamp(th, min_C+1e-3, max_C-1e-3)


class SmoothThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, th, sharpness=10):
        th = clamp_threshold(C,th)
        ctx.save_for_backward(C, th)
        ctx.sharpness = sharpness
        logistic = 1 / (1 + torch.exp(-sharpness * (C[:, 0] - th)))
        return torch.round(logistic)

    @staticmethod
    def backward(ctx, grad_output):
        C, th = ctx.saved_tensors
        sharpness = ctx.sharpness
        sigmoid_derivative = sharpness * torch.exp(-sharpness * (C[:, 0] - th)) / (1 + torch.exp(-sharpness * (C[:, 0] - th)))**2
        grad_C = grad_output[:, None] * sigmoid_derivative[:, None] * C[:, 1:] 
        grad_th = -torch.sum(grad_output * sigmoid_derivative * C[:, 1])
        return grad_C, grad_th, None

def threshold(C, th, sharpness = 10):
    return SmoothThresholdFunction.apply(C, th, sharpness)
    
def amplitudes(A, P, gamma, beta):
    # A, amplitudes of basis state, shape: (N,) N is number of values of the phase function
    # P, phase function values, shape: (N,2), the 1st column represents values, the 2nd one represents counts of correspondiong values
    # if using GM-Th-QAOA, N = 2
    num = torch.sum(P[:,1])
    left = -((1-torch.exp(-1j*beta))/num)*torch.sum(A * torch.exp(-1j*gamma*P[:,0]) * P[:,1])
    right = A * torch.exp(-1j*gamma*P[:,0])
    return left+right
    # return torch.where(P[:,1] == 0, torch.tensor(0.0, dtype=A.dtype), left+right)

def process_threshold(C, gammas, betas, th, sharpness = 10):
    num = torch.sum(C[:,1])
    # amplitudes of states corresponding to phase value 1 and 0
    A = torch.tensor([1/torch.sqrt(num), 1/torch.sqrt(num)], device=C.device, dtype=torch.complex128)
    # phase values
    P = torch.zeros(2, 2, device=C.device)
    P[0, 0] = 1
    P[1, 0] = 0
    threshold_result = threshold(C, th, sharpness)
    P[0, 1] = torch.sum(C[:, 1] * (1 - threshold_result))
    P[1, 1] = torch.sum(C[:, 1] * threshold_result)
    
    rounds = len(gammas)
    for p in range(rounds):
        A = amplitudes(A, P, gammas[p], betas[p])
    return A

def process(C, gammas, betas):
    num = torch.sum(C[:,1])
    A = torch.tensor([1/torch.sqrt(num)] * C.shape[0], device=C.device, dtype=torch.complex128)
    rounds = len(gammas)
    for p in range(rounds):
        A = amplitudes(A, C, gammas[p], betas[p])
    return A

def expectation_threshold(A, C, th, sharpness = 10):
    # A, amplitudes of basis state, shape: (2,)
    # C, objective function values, shape: (N,2)
    threshold_result = threshold(C, th, sharpness)
    return torch.sum(torch.abs(A[0])**2 * C[:,0] * C[:,1] * (1 - threshold_result) + 
                     torch.abs(A[1])**2 * C[:,0] * C[:,1] * threshold_result)

def expectation(A,C):
    # A, amplitudes of basis state, shape: (N,)
    # C, objective function values, shape: (N,2), the 1st column represents values, the 2nd one represents counts of correspondiong values
    return torch.sum(torch.abs(A)**2 * C[:,0] * C[:,1])

def optimal_prob_threshold(A,C):
    opt_idx = torch.argmin(C[:,0])
    opt_count = C[opt_idx,1]
    return opt_count * torch.abs(A[0])**2
    
def optimal_prob(A,C):
    opt_idx = torch.argmin(C[:,0])
    opt_count = C[opt_idx,1]
    return opt_count * torch.abs(A[opt_idx])**2


class GMQAOA:
    def __init__(self, rounds, distribution, th_strategy = True, opt_type = 'max' ,target_metric = 'p_opt', initial_gammas = [], initial_betas = [], sharpness = 1, initial_th = None, display = True):
        # distribution, objective function values, shape: (N,2)
        self.device = distribution.device
        
        self.rounds = rounds
        
        self.th_strategy = th_strategy
        self.sharpness = sharpness
        
        self.C = distribution.clone()
        if opt_type == 'max':
            self.C[:, 0] = -self.C[:, 0]
            
        self.opt_count = self.C[torch.argmin(self.C[:,0]),1]
        self.feasible_count = torch.sum(self.C[:,1])
        self.rho = self.opt_count/self.feasible_count
            
        # paramaters
        if self.th_strategy:
            if initial_th != None:
                self.th = torch.tensor(initial_th, device=self.device, requires_grad=True, dtype=torch.float64)
            else:
                mean = torch.sum(self.C[:,0]*self.C[:,1])/torch.sum(self.C[:,1])
                opt = torch.min(self.C[:,0])
                self.th = torch.tensor((opt.item()+mean.item())/2, device=self.device, requires_grad=True, dtype=torch.float64)

            # self.optUb = poptUbTh(self.rho,self.rounds)
            
        else:
            self.th = None

        self.optUb = ((2*self.rounds+1)**2) * self.rho
        
        if initial_gammas == [] or initial_betas == []:
            self.gammas = torch.randn(self.rounds, device=self.device, requires_grad=True, dtype=torch.float64)
            self.betas = torch.randn(self.rounds, device=self.device, requires_grad=True, dtype=torch.float64)
        else:
            self.gammas = torch.tensor(initial_gammas, device=self.device, requires_grad=True, dtype=torch.float64)
            self.betas = torch.tensor(initial_betas, device=self.device, requires_grad=True, dtype=torch.float64)
            
        self.target_metric = target_metric

        self.display = display
        
    def get_amplitudes(self,gammas,betas,th=None):
        if th is not None: 
            A = process_threshold(self.C, gammas, betas, th, self.sharpness)
        else:
            A = process(self.C, gammas, betas)
        return A
    
    def grid_search(self, prev_gammas, prev_betas, prev_th = None):
        if prev_th != None:
            prev_th = torch.tensor(prev_th, device=self.device, dtype=torch.float64)
        if len(prev_gammas) == self.rounds-1 and len(prev_betas) == self.rounds-1:
            best = torch.inf
            for gamma in np.linspace(0,2*torch.pi,100):
                for beta in np.linspace(0,2*torch.pi,30):
                    gamma_tensor = torch.tensor(prev_gammas+[gamma], device=self.device, dtype=torch.float64)
                    beta_tensor = torch.tensor(prev_betas+[beta], device=self.device, dtype=torch.float64)
                    A = self.get_amplitudes(gamma_tensor,beta_tensor,prev_th)
                    if self.target_metric == 'p_opt':
                        if self.th_strategy:
                            metric = -optimal_prob_threshold(A, self.C)
                        else:
                            metric = -optimal_prob(A, self.C)
                    else:
                        if self.th_strategy:
                            exp = expectation_threshold(A, self.C, prev_th, self.sharpness)
                        else:
                            exp = expectation(A, self.C)
                        metric = exp/torch.abs(torch.min(self.C[:,0]))
                    if metric.item() < best:
                        best = metric.item()
                        self.gammas = gamma_tensor.clone()
                        self.betas = beta_tensor.clone()
                        
        self.gammas.requires_grad_(True)
        self.betas.requires_grad_(True)
    
    def initializing(self,lr=2e-2, max_step = 200):
        if self.rounds == 1 and self.th_strategy == False:
            self.grid_search([],[],None)
            
        else:
            T = torch.tensor([1.], device=self.device, requires_grad=True)
            
            if self.th_strategy:
                min_diff = torch.min(torch.sort(self.C[:,0]).values.diff())
                optimizer = torch.optim.Adam([
                    {'params': [T], 'lr': lr},
                    {'params': [self.th], 'lr': min_diff/5}
                ])
            else:
                optimizer = torch.optim.Adam([T], lr)
            start_time = time.time()
            for step in range(max_step):
                optimizer.zero_grad()
                A = self.get_amplitudes(torch.arange(1, self.rounds + 1, device=self.device).float() * (T / self.rounds) / self.rounds, (1 - torch.arange(1, self.rounds + 1, device=self.device).float()/self.rounds) * (T / self.rounds), self.th)
                if self.target_metric == 'p_opt':
                    if self.th_strategy:
                        metric = -optimal_prob_threshold(A, self.C)
                    else:
                        metric = -optimal_prob(A, self.C)
                else:
                    if self.th_strategy:
                        exp = expectation_threshold(A, self.C, self.th, self.sharpness)
                    else:
                        exp = expectation(A, self.C)
                    metric = exp/torch.abs(torch.min(self.C[:,0]))

                if step % (max_step//20) == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    if self.display:
                        print(f"TQA: Level {self.rounds}, {self.target_metric}, Step {step+1}, loss={metric.item()}, time: {elapsed_time:.6f}s",flush=True)
                    
                metric.backward()
                optimizer.step()
                
            self.gammas = torch.arange(1, self.rounds + 1, device=self.device).float() * (T.item() / self.rounds) / self.rounds
            self.betas = (1 - torch.arange(1, self.rounds + 1, device=self.device).float()/self.rounds) * (T.item() / self.rounds)
            
            self.gammas.requires_grad_(True)
            self.betas.requires_grad_(True)
    
    def run(self, lr= 5e-3, max_step = 1200):
        
        if self.th_strategy:
            min_diff = torch.min(torch.sort(self.C[:,0]).values.diff())
            optimizer = torch.optim.Adam([
                {'params': [self.gammas, self.betas], 'lr': lr},
                {'params': [self.th], 'lr': min_diff/3}
            ])
        else:
            optimizer = torch.optim.Adam([self.gammas,self.betas], lr)
        
        best = torch.inf
        
        start_time = time.time()
        
        for step in range(max_step):
            optimizer.zero_grad()
            
            A = self.get_amplitudes(self.gammas,self.betas,self.th)
            if self.target_metric == 'p_opt':
                if self.th_strategy:
                    metric = -optimal_prob_threshold(A, self.C)
                else:
                    metric = -optimal_prob(A, self.C)
            else:
                if self.th_strategy:
                    exp = expectation_threshold(A, self.C, self.th, self.sharpness)
                else:
                    exp = expectation(A, self.C)
                metric = exp/torch.abs(torch.min(self.C[:,0]))
            
            # save the best parameters
            if metric.item() < best:
                best = metric.item()
                self.opt_gammas = self.gammas.clone()
                self.opt_betas = self.betas.clone()
                if self.th_strategy:
                    self.opt_th = clamp_threshold(self.C,self.th.clone())
                else:
                    self.opt_th = None

            if -np.round(metric.item(),5) == 1:
                break
            
            if self.target_metric == 'p_opt':
                metric = metric/self.optUb

            if step % (max_step//10) == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                if self.display:
                    print(f"Level {self.rounds}, {self.target_metric}, Step {step+1}, loss={metric.item()}, time: {elapsed_time:.6f}s",flush=True)
                    
            metric.backward()
            
            optimizer.step()
        
        A = self.get_amplitudes(self.opt_gammas,self.opt_betas,self.opt_th)
        if self.th_strategy:
            self.p_opt_value = optimal_prob_threshold(A, self.C)
            self.approx_value = expectation_threshold(A, self.C, self.opt_th, self.sharpness)/torch.min(self.C[:,0])
        else:
            self.p_opt_value = optimal_prob(A,self.C)
            self.approx_value = expectation(A, self.C)/torch.min(self.C[:,0])
        self.opt_gap_value = self.approx_value - 1
                