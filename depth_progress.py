from utils import get_save_distribtuion
from gmqaoa import GMQAOA

import json,os
import numpy as np
import torch
import sys


STRATEGYS = [{'lr':5e-3, 'max_step':1280, 'initializing':'NO'},
             {'lr':5e-3, 'max_step':1280, 'initializing':'TQA'},
             {'lr':1e-2, 'max_step':960, 'initializing':'NO'},
             {'lr':1e-2, 'max_step':960, 'initializing':'TQA'},
             {'lr':2e-2, 'max_step':720, 'initializing':'NO'},
             {'lr':2e-2, 'max_step':720, 'initializing':'TQA'}]

STRATEGYS_TH = [{'lr':2e-3, 'max_step':2400, 'initializing':'Prev'},
                {'lr':1e-3, 'max_step':2400, 'initializing':'NO'},
                {'lr':2e-2, 'max_step':1800, 'initializing':'TQA'},
                {'lr':2e-2, 'max_step':1800, 'initializing':'NO'},
                {'lr':5e-3, 'max_step':2200, 'initializing':'NO'},
                {'lr':5e-3, 'max_step':2200, 'initializing':'TQA'},
                {'lr':2e-2, 'max_step':1800, 'initializing':'Equal'}]


def optimizing(_type, distribution, level, th_strategy, initial_gammas = [], initial_betas = [], initial_th = None, lr = 5e-3, max_step = 1200, initializing = True, sharpness = 1):
    if _type == 'approx':
        gq = GMQAOA(level, torch.from_numpy(distribution).to('cpu'), th_strategy = th_strategy, opt_type='max', target_metric='approx',initial_gammas = initial_gammas, initial_betas = initial_betas, sharpness = sharpness, initial_th = initial_th)
    elif _type == 'gap':
        gq = GMQAOA(level, torch.from_numpy(distribution).to('cpu'), th_strategy = th_strategy, opt_type='min', target_metric='gap',initial_gammas = initial_gammas, initial_betas = initial_betas, sharpness = sharpness, initial_th = initial_th)
    elif _type == 'max':
        gq = GMQAOA(level, torch.from_numpy(distribution).to('cpu'), th_strategy = th_strategy, opt_type='max', target_metric='p_opt',initial_gammas = initial_gammas, initial_betas = initial_betas, sharpness = sharpness, initial_th = initial_th)
    elif _type == 'min':
        gq = GMQAOA(level, torch.from_numpy(distribution).to('cpu'), th_strategy = th_strategy, opt_type='min', target_metric='p_opt',initial_gammas = initial_gammas, initial_betas = initial_betas, sharpness = sharpness, initial_th = initial_th)
    if initializing:
        gq.initializing(lr=2e-2,max_step = 100)
    gq.run(lr,max_step)
    gammas = gq.opt_gammas.tolist()
    betas = gq.opt_betas.tolist()
    th = gq.opt_th
    p_opt = gq.p_opt_value.item()
    approx_ratio = gq.approx_value.item()
    opt_gap = gq.opt_gap_value.item()
    tts = 1/p_opt
    return gammas,betas,th,p_opt,approx_ratio,opt_gap,tts

def save(opt_value,
         opt_count,
         feasible_count,
         mean_value,
         std,
         level,
         gammas,
         betas,
         predicted_p_opt,
         predicted_tts,
         p_opt,
         tts,
         approx_ratio,
         opt_gap,
         th,
         th_strategy,
         problem_dict,
         save_path):
    save_dict = {
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
    if th_strategy:
        save_dict['th'] = th.item()
    problem_dict.update(save_dict)
    save_dict = problem_dict
    save_dict = {k: int(v) if isinstance(v, np.integer) else v for k, v in save_dict.items()}
    save_dict = {k: float(v) if isinstance(v, np.float32) else v for k, v in save_dict.items()}
    with open(save_path, 'w') as f:
        json.dump(save_dict, f)

def read_json(save_path,_type):
    with open(save_path, "r") as json_file:
        save_dict = json.load(json_file)
    if _type == 'approx':
        base = save_dict['approx']
    elif _type == 'gap':
        base = -save_dict['gap']
    else:
        base = save_dict['p_opt']
    return base, save_dict['gammas'], save_dict['betas'], save_dict.get('th',None)
    

# get approx and save result
def depth_progress_opt(params,_type,get_problem,get_distribtuion_path,get_distribtuion,get_result_path,levels = [i for i in range(1,50,2)],th_strategy = False):
    
    problem_name = "_".join([str(i) for i in params])
    sys.stdout.flush()
        
    problem_dict = get_problem(*params)
    problem = [problem_dict[key] for key in problem_dict.keys()]

    print(problem_name,f' start!')
    sys.stdout.flush()
                        
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

    level = levels[0]

    save_path = get_result_path(*params+[level])

    if not os.path.exists(save_path):
    
        predicted_p_opt = opt_count*((2*level+1)**2)/feasible_count

        initial_gammas = []
        initial_betas = []
        initializing = True
        
        if level == 1:
            if th_strategy:
                initial_gammas = np.array([3.14]*level) - np.random.rand(level) * 0.2
                initial_betas = np.array([3.14]*level) - np.random.rand(level) * 0.2
                initial_gammas = initial_gammas.tolist()
                initial_betas = initial_betas.tolist()
                initializing = False
            else:
                initializing = False

        initial_th = None
        if th_strategy:
            if _type == 'min':
                initial_th = np.min(distribution[:, 0]) + 1e-3
            elif _type == 'max':
                initial_th = np.min(-distribution[:, 0]) + 1e-3
             
        gammas,betas,th,p_opt,approx_ratio,opt_gap,tts = optimizing(_type, distribution, level, th_strategy, initial_gammas, initial_betas, initial_th, 5e-3, 1200, initializing)

        if level == 1 and th_strategy == False:
            for _ in range(6):
                gammas_tmp,betas_tmp,th_tmp,p_opt_tmp,approx_ratio_tmp,opt_gap_tmp,tts_tmp = optimizing(_type, distribution, level, th_strategy, initial_gammas, initial_betas, initial_th, 5e-3, 1200, initializing)
                update = False
                if _type == 'approx':
                    if approx_ratio_tmp > approx_ratio:
                        update = True
                elif _type == 'gap':
                    if opt_gap_tmp < opt_gap:
                        update = True
                else:
                    if p_opt_tmp > p_opt:
                        update = True
                if update:
                    gammas,betas,th,p_opt,approx_ratio,opt_gap,tts = gammas_tmp,betas_tmp,th_tmp,p_opt_tmp,approx_ratio_tmp,opt_gap_tmp,tts_tmp
        
        save(opt_value,opt_count,feasible_count,mean_value,std,level,gammas,betas,predicted_p_opt,1/predicted_p_opt,
             p_opt,tts,approx_ratio,opt_gap,th,th_strategy,problem_dict,save_path)

    base, prev_gammas, prev_betas, prev_th = read_json(save_path,_type)
                    
    print(problem_name,f'depth-{level} Done!' + '\n' + f'{_type}: {base}')
    sys.stdout.flush()

    for level in levels[1:]:
        if base >= 0.99:
            break
        
        predicted_p_opt = opt_count*((2*level+1)**2)/feasible_count
        
        save_path = get_result_path(*params+[level])

        optimize = True
        if os.path.exists(save_path):
            base_new, _, _, _ = read_json(save_path,_type)
            if base_new > base or np.round(base_new,5)==1:
                optimize = False
        
        
        if optimize:
            
            results = []
            bases = []
            
            if th_strategy:
                strategies = STRATEGYS_TH
            else:
                strategies =  STRATEGYS
                

            initial_gammas = np.array([3.14]*level) - np.random.rand(level) * 0.2
            initial_betas = np.array([3.14]*level) - np.random.rand(level) * 0.2
            
            gq_tmp = GMQAOA(level, torch.from_numpy(distribution).to('cpu'), th_strategy = th_strategy, opt_type='max', target_metric='approx',initial_gammas = [], initial_betas = [], display = False)
            gq_tmp.grid_search(prev_gammas,prev_betas,prev_th)
            grid_gammas = gq_tmp.gammas.detach().cpu().numpy().tolist()
            grid_betas = gq_tmp.betas.detach().cpu().numpy().tolist()

            for strategy in strategies:
                if strategy['initializing'] == 'TQA':
                    initializing = True
                    initial_gammas = []
                    initial_betas = []
                elif strategy['initializing'] == 'Prev':
                    initializing = False
                    additional_gammas = np.array([np.mean(prev_gammas)]*(level-len(prev_gammas))) - np.random.rand(level-len(prev_gammas)) * 0.2
                    additional_betas = np.array([np.mean(prev_betas)]*(level-len(prev_betas))) - np.random.rand(level-len(prev_betas)) * 0.2
                    initial_gammas = prev_gammas + additional_gammas.tolist()
                    initial_betas = prev_betas + additional_betas.tolist()
                elif strategy['initializing'] == 'Equal':
                    initializing = False
                    initial_gammas = np.array([3.14]*level) - np.random.rand(level) * 0.1
                    initial_betas = np.array([3.14]*level) - np.random.rand(level) * 0.1
                    initial_gammas = initial_gammas.tolist()
                    initial_betas = initial_betas.tolist()
                elif strategy['initializing'] == 'NO':
                    initializing = False
                    initial_gammas = grid_gammas
                    initial_betas = grid_betas
                gammas,betas,th,p_opt,approx_ratio,opt_gap,tts = optimizing(_type, distribution, level, th_strategy, initial_gammas, initial_betas, prev_th, strategy['lr'], strategy['max_step'], initializing)

                results.append([gammas,betas,th,p_opt,approx_ratio,opt_gap,tts])
                
                if _type == 'approx':
                    base_new = approx_ratio
                elif _type == 'gap':
                    base_new = -opt_gap
                else:
                    base_new = p_opt

                bases.append(base_new)
                    
            best_idx = np.argmax(bases)
            gammas,betas,th,p_opt,approx_ratio,opt_gap,tts = results[best_idx]
            
            save(opt_value,opt_count,feasible_count,mean_value,std,level,gammas,betas,predicted_p_opt,1/predicted_p_opt,
                         p_opt,tts,approx_ratio,opt_gap,th,th_strategy,problem_dict,save_path)
            
        base_new, prev_gammas, prev_betas, prev_th = read_json(save_path,_type)

        if np.round(base_new,3) < np.round(base,3) and np.round(base_new,5) != 1:
            if np.abs(base_new-base) > 1e-2:
                print(problem_name,f' level-{level} Optimizing Failed!!!!!!!!!!')
                break
            else:
                base = base_new
                
        else:
            base = base_new

        print(problem_name,f'depth-{level} Done!' + '\n' + f'{_type}: {base}')
        sys.stdout.flush()

    print(problem_name,f' Finished')
            

