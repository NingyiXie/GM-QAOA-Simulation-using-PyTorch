from utils import get_save_distribtuion
from gmqaoa import GMQAOA

import json,os
import numpy as np
import torch
import sys


# get approx and save result
def sample_gap_approx_opt(params,_type,get_problem,get_distribtuion_path,get_distribtuion,get_result_path):
    level = params[-1]
    params = params[:-1]
    
    problem_name = "_".join([str(i) for i in params])
    sys.stdout.flush()
        
        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        #     # print("cuda is available")
        # else:
        #     device = torch.device("cpu")
        #     # print("use cpus")
        # sys.stdout.flush()
    device = torch.device("cpu")
    
    problem_dict = get_problem(*params)
    problem = [problem_dict[key] for key in problem_dict.keys()]
        
    if level != 0:
        save_path = get_result_path(*params+[level])
            
        if not os.path.exists(save_path):
            print(problem_name,f'depth-{level} start!')
            
            distribution = get_save_distribtuion(problem,get_distribtuion_path,get_distribtuion,params)
                
            if _type == 'approx' or _type == 'max':
                opt_value = np.max(distribution[:, 0]) # optimal objective value
                opt_count = distribution[np.argmax(distribution[:, 0]),1] # the number of solutions corresponding to the opt_value
            elif _type == 'gap' or _type == 'min':
                opt_value = np.min(distribution[:, 0]) # optimal objective value
                opt_count = distribution[np.argmin(distribution[:, 0]),1] # the number of solutions corresponding to the opt_value

            feasible_count = np.sum(distribution[:, 1]) # the number of total solutions
            mean_value = np.sum(distribution[:, 0] * distribution[:, 1])/feasible_count # avg
            std = np.sqrt(np.sum(distribution[:, 1] * ((distribution[:, 0] - mean_value)**2))/feasible_count) # std
                
            predicted_p_opt = opt_count*((2*level+1)**2)/feasible_count
            predicted_tts = 1/predicted_p_opt
                
            if _type == 'approx':
                gq = GMQAOA(level,torch.tensor(distribution,device = device), opt_type='max', target_metric='approx')
            elif _type == 'gap':
                gq = GMQAOA(level,torch.tensor(distribution,device = device), opt_type='min', target_metric='gap')
            elif _type == 'max':
                gq = GMQAOA(level,torch.tensor(distribution,device = device), opt_type='max', target_metric='p_opt')
            elif _type == 'min':
                gq = GMQAOA(level,torch.tensor(distribution,device = device), opt_type='min', target_metric='p_opt')
                       
            if level!=1:
                gq.initializing()
            gq.run()
            gammas = gq.gammas.tolist()
            betas = gq.betas.tolist()
            p_opt = gq.p_opt_value
            approx_ratio = gq.approx_value
            opt_gap = gq.opt_gap_value
            tts = 1/p_opt

            save_dict = {
                    "distribution": distribution.tolist(),
                    "opt_value": opt_value,
                    "opt_count": opt_count,
                    "feasible_count": feasible_count,
                    "mean_value": mean_value,
                    "std": std,
                    "level": level,
                    "gammas": gammas,
                    "betas": betas,
                    "predicted_p_opt": predicted_p_opt,
                    "predicted_tts": predicted_tts,
                    "p_opt": p_opt,
                    "tts": tts,
                    "approx":approx_ratio,
                    "opt_gap":opt_gap
            }
            problem_dict.update(save_dict)
            save_dict = problem_dict
            save_dict = {k: int(v) if isinstance(v, np.integer) else v for k, v in save_dict.items()}
            save_dict = {k: float(v) if isinstance(v, np.float32) else v for k, v in save_dict.items()}
            with open(save_path, 'w') as f:
                json.dump(save_dict, f)
                
            if _type == 'approx':
                info = f'approx: {approx_ratio}'
            elif _type == 'gap':
                info = f'gap: {opt_gap}'
            else:
                info = f'p_opt: {p_opt}'
                    
            print(problem_name,f'depth-{level} Done!' + '\n' + info)
            sys.stdout.flush()
                
        else:
            print(problem_name,f'depth-{level} exists')
            sys.stdout.flush()
    else:
        get_save_distribtuion(problem,get_distribtuion_path,get_distribtuion,params)
        print(problem_name,'Done!')
