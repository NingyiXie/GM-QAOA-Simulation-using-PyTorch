from gmqaoa import GMQAOA
from fractions import Fraction
import multiprocessing
import argparse
import torch
import numpy as np
import pandas as pd
import json,os

STRATEGYS = [{'lr':1e-3, 'max_step':2560, 'initializing':'NO'},
             {'lr':1e-3, 'max_step':2560, 'initializing':'TQA'},
             {'lr':5e-3, 'max_step':1280, 'initializing':'NO'},
             {'lr':5e-3, 'max_step':1280, 'initializing':'TQA'},
             {'lr':1e-2, 'max_step':960, 'initializing':'NO'},
             {'lr':1e-2, 'max_step':960, 'initializing':'TQA'},
             {'lr':2e-2, 'max_step':720, 'initializing':'NO'},
             {'lr':2e-2, 'max_step':720, 'initializing':'TQA'}]

def stdOverOptMinusMean(dist,opt_type='max'): # example: dist = np.array([[3,1],[1,49],[0,950]])
    dist_copy = dist.copy()
    if opt_type == 'max':
        opt = np.max(dist[:,0])
        dist_copy[np.argmax(dist[:,0]),1] = 0
    elif opt_type == 'min':
        opt = np.min(dist[:,0])
        dist_copy[np.argmin(dist[:,0]),1] = 0
    mean = np.sum(dist_copy[:, 0] * dist_copy[:, 1])/np.sum(dist_copy[:, 1])
    std = np.sqrt(np.sum(dist_copy[:, 1] * ((dist_copy[:, 0] - mean)**2))/np.sum(dist_copy[:, 1]))
    return std/np.abs(opt-mean)

def worker(params):
    idx, max_p = params
    
    save_path = f'./popt_result/random/{idx}.json'
    
    if not os.path.exists(save_path):
    
        dist = np.load(f'./distribution/random/{idx}.npy')
        opt_count = dist[np.argmax(dist[:,0]),1]
        feasible_count = np.sum(dist[:,1])
        opt_value = np.max(dist[:,0])
        opt_density = opt_count/feasible_count
        stdoptmean = stdOverOptMinusMean(dist)
        
        results = []
        
        if len(dist[:,0]) == 2:
            gq = GMQAOA(1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = [3.14], initial_betas = [3.14], display = False)
            gq.run(2e-2,1000)
        else:
            gq = GMQAOA(1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = [], initial_betas = [], display = False)
            gq.initializing()
            gq.run(5e-3,1200)
        
        base = gq.p_opt_value.item()
        prev_gammas = gq.opt_gammas.detach().cpu().numpy().tolist()
        prev_betas = gq.opt_betas.detach().cpu().numpy().tolist()
    
        results.append({'level':1,'gammas':prev_gammas,'betas':prev_betas,'p_opt':base})
        
        for p in range(1,max_p):
            if base >= 0.99:
                break
    
            initial_gammas = np.array([3.14]*(p+1)) + np.random.rand(p+1)*0.01-0.005
            initial_betas = np.array([3.14]*(p+1)) + np.random.rand(p+1)*0.01-0.005
            gq = GMQAOA(p+1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = initial_gammas.tolist(), initial_betas = initial_betas.tolist(), display = False)
            gq.run(2e-2,300)
    
            if len(dist[:,0]) == 2:
                gq_try = GMQAOA(p+1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = [], initial_betas = [], display = False)
                gq_try.initializing()
                gq_try.run(2e-2,1000)
                if gq.p_opt_value.item() < gq_try.p_opt_value.item():
                    gq = gq_try
                initial_gammas = np.array([3.14]*(p+1)) + np.random.rand(p+1)*0.01-0.005
                initial_betas = np.array([3.14]*(p+1)) + np.random.rand(p+1)*0.01-0.005
                gq_try = GMQAOA(p+1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = initial_gammas.tolist(), initial_betas = initial_betas.tolist(), display = False)
                gq_try.run(5e-3,1200)
                if gq.p_opt_value.item() < gq_try.p_opt_value.item():
                    gq = gq_try
            else:
                gq_tmp = GMQAOA(p+1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = [], initial_betas = [], display = False)
                gq_tmp.grid_search(prev_gammas,prev_betas)
                initial_gammas = gq_tmp.gammas.detach().cpu().numpy().tolist()
                initial_betas = gq_tmp.betas.detach().cpu().numpy().tolist()
                
                for strategy in STRATEGYS:
                    gq_try = GMQAOA(p+1, torch.from_numpy(dist).to('cpu'), th_strategy = False, opt_type='max', target_metric='p_opt',initial_gammas = initial_gammas, initial_betas = initial_betas, display = False)
                    if strategy['initializing'] == 'TQA':
                        gq_try.initializing()
                    gq_try.run(strategy['lr'],strategy['max_step'])
                    if gq.p_opt_value.item() < gq_try.p_opt_value.item():
                        gq = gq_try
                        
            base_new = gq.p_opt_value.item()
            prev_gammas = gq.opt_gammas.detach().cpu().numpy().tolist()
            prev_betas = gq.opt_betas.detach().cpu().numpy().tolist()
            if base_new > base:
                base = base_new
                print(f'{idx} depth-{p+1}: {base}',flush=True)
                results.append({'level':p+1,'gammas':prev_gammas,'betas':prev_betas,'p_opt':base})
                save_dict = {
                    'opt_value':opt_value,
                    'opt_count':opt_count,
                    'feasible_count':feasible_count,
                    'opt_density':opt_density,
                    'stdOverOptMinusMean':stdoptmean,
                    'results':results
                }
                
                with open(save_path, 'w') as f:
                    json.dump(save_dict, f)
            else:
                print(f'{idx} Failed at depth-{p+1}',flush=True)
                break
                
        print(f'{idx} Finished',flush=True)

    else:
        print(f'{idx} Exists',flush=True)
    

def main():
    max_p = 50
    args_list = [[i,max_p] for i in range(150)]
    num_processors = min([48, multiprocessing.cpu_count()])
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_processors) as pool:
        pool.map(worker, args_list)
    pool.close()
    pool.join()



if __name__ == '__main__':
    main()
