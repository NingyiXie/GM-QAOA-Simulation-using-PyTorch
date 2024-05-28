from gmqaoa import GMQAOA

import numpy as np
from fractions import Fraction
import torch

from tqdm import tqdm

def create_distribution(r):
    # Ensure r is within the valid range [0, 1]
    if not (0 <= r <= 1):
        return "r must be within the interval [0, 1]."

    # Convert r to its simplest fraction form
    frac = Fraction(r).limit_denominator()
    a11 = frac.numerator
    a01 = frac.denominator - a11  # Since we need a01 + a11 to be as small as possible

    # Construct the 2x2 numpy array with specified conditions
    a = np.array([[0, a01], [1, a11]])
    return a

def get_r_vs_p_opt(p,rs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print("cuda is available")
    else:
        device = torch.device("cpu")
        # print("use cpus")
        
    results = np.zeros((len(rs),3+2*p))
    
    for idx in tqdm(range(len(rs))):
        r = rs[idx]
        
        if r != 1:
            dist = torch.tensor(create_distribution(r), device=device)
            popt = GMQAOA(p,dist,target_metric='p_opt')
            popt.initializing()
            popt.run()
            gammas = popt.gammas.tolist()
            betas = popt.betas.tolist()
            p_opt = popt.p_opt_value
            if p_opt>1:
                p_opt==1
        elif r == 1:
            gammas = [0]*p
            betas = [0]*p
            p_opt = 1
            
        results[idx] = [r]+[p_opt]+[p_opt/r]+gammas+betas
    
    return results


if __name__ == "__main__":
    rs = np.linspace(0,1,10001)[1:]
    p = 2
    results = get_r_vs_p_opt(p,rs)
    np.save('results.npy', sorted_results)