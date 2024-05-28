import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import json
import os

from multiprocessing import Pool

from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def get_save_distribtuion(problem,get_distribtuion_path,compute_distribtuion,params):
    dist_path = get_distribtuion_path(*params)
    
    if not os.path.exists(dist_path):
        distribution = compute_distribtuion(*problem)
        np.save(dist_path,distribution)
    else:
        print('exists')
    
    return np.load(dist_path)


def sum_of_top_n_values(dist, n): # sum_{i=1}^{r * |F|} sort(D)
    dist_sorted = dist[dist[:, 0].argsort()[::-1]]
    
    total_sum = 0
    elements_counted = 0
    
    for value, count in dist_sorted:
        if elements_counted + count <= n:
            total_sum += value * count
            elements_counted += count
        else:
            total_sum += value * (n - elements_counted)
            break
    
    return total_sum

def get_approxUB(r,dist):
    counts = np.sum(dist[:,1])
    n = math.ceil(r*counts)
    return sum_of_top_n_values(dist,n)/(r*counts*np.max(dist[:,0]))


def sum_of_bottom_n_values(dist, n): # sum_{i=1}^{r * |F|} sort(D)
    dist_sorted = dist[dist[:, 0].argsort()]
    
    total_sum = 0
    elements_counted = 0
    
    for value, count in dist_sorted:
        if elements_counted + count <= n:
            total_sum += value * count
            elements_counted += count
        else:
            total_sum += value * (n - elements_counted)
            break
    
    return total_sum


def get_gapLB(r,dist):
    counts = np.sum(dist[:,1])
    n = math.floor(r*counts)
    return sum_of_bottom_n_values(dist,n)/(n*np.min(dist[:,0]))-1


def get_p_opt_UB(p,dist,opt='max'):
    if opt == 'max':
        opt_count = dist[np.argmax(dist[:, 0]),1]
    elif opt == 'min':
        opt_count = dist[np.argmin(dist[:, 0]),1]
    feasible_count = np.sum(dist[:, 1])
    return ((2*p+1)**2)*opt_count/feasible_count



def read_single_data(args):
    N, idx, get_path, r_array, log_inv_r_array, metric, opt = args
    dist = np.load(get_path(N, idx))  # read numpy
    
    opt_density = get_p_opt_UB(0,dist,opt)
    
    results = {
        'problem': [f'n{N}_idx{idx}'] * len(r_array),
        'N': [N] * len(r_array),
        'r': r_array.tolist(),
        'log(1/r)': log_inv_r_array.tolist(),
        'opt_density': [opt_density] * len(r_array),
    }
    if metric == 'approxUB':
        results[metric] = [get_approxUB(r, dist) for r in r_array]
    elif metric == 'gapLB':
        results[metric] = [get_gapLB(r, dist) for r in r_array]        
    return results


def read_data(N_list, idx_list, get_path, metric='approxUB', opt ='max', num_processes=4, resolution = 800):
    log_inv_r_array = np.linspace(0, 8.8, resolution)
    if resolution == 1:
        log_inv_r_array = np.array([0])
    r_array = 1 / np.exp(log_inv_r_array)
    
    # r_array = np.linspace(0, 1, 1001)[1:]
    # log_inv_r_array = np.log(1/r_array)
    
    task_list = [(N, idx, get_path, r_array, log_inv_r_array, metric, opt) for N in N_list for idx in idx_list]
    
    with Pool(processes=num_processes) as pool:
        results_list = list(tqdm(pool.imap(read_single_data, task_list), total=len(task_list)))

    # Consolidate results
    consolidated_results = {key: [] for key in results_list[0]}
    for result in results_list:
        for key in result:
            consolidated_results[key].extend(result[key])

    df = pd.DataFrame(consolidated_results)
    return df


def read_sampled_data(N_list,idx_list,p_list,get_path, metric = 'approx', get_dist_path = None, model = None, params = None, polymodel=None):
    data_tag = metric

    if metric == 'max' or metric == 'min':
        data_tag = 'p_opt'
        
    results = {'problem':[],'N':[],'depth':[],data_tag:[]}
    if get_dist_path != None:
        results[data_tag+'Bound'] = []
        if model != None:
            results[data_tag+'FittedBound'] = []
        if polymodel != None:
            results[data_tag+'PolyBound'] = []
    
    pbar = tqdm(total=len(N_list)*len(idx_list))
    for N in N_list:
        for idx in idx_list:
            for p in p_list:
                file_path = get_path(N,p,idx)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                results['problem'].append(f'n{N}_idx{idx}')
                results['N'].append(N)
                results['depth'].append(p)
                results[data_tag].append(data[data_tag])
                if get_dist_path != None:
                    dist = np.load(get_dist_path(N, idx))
                    if data_tag == 'approx':
                        results[data_tag+'Bound'].append(get_approxUB(1/((2*p+1)**2), dist))
                    elif data_tag == 'opt_gap':
                        results[data_tag+'Bound'].append(get_gapLB(1/((2*p+1)**2), dist))
                    elif metric == 'max' or metric == 'min':
                        results[data_tag+'Bound'].append(get_p_opt_UB(p,dist,opt=metric))
                    if model != None:
                        results[data_tag+'FittedBound'].append(model(N,p,params))
                    if polymodel != None:
                        results[data_tag+'PolyBound'].append(polymodel(np.array(dist,N,p)))
                pbar.update(1)
    pbar.close()
    df = pd.DataFrame(results)
    return df


def fit_model(data):
    def model(N,slope,intercept):
        return slope*N+intercept
    # Extract data for N, log1r, approx
    N_data = data['N'].values
    log_density_data = data['log(opt_density)'].values
    density_data = data['opt_density'].values
    
    # Initial guesses for the parameters
    initial_guesses = [-0.1, 0.1]
    
    # Curve fitting
    params, covariance = curve_fit(model, N_data, log_density_data, p0=initial_guesses)

    # Calculate predictions and mean squared error
    predicted_log_density = np.exp(model(N_data, *params))
    mse = mean_squared_error(density_data, predicted_log_density)
    
    return params,mse

def optimized_model(p,n,params):
    return ((2*p+1)**2)*np.exp(params[0]*n+params[1])


def fitted_mu(N,p,params):
    if len(params) == 3:
        slope,intercept,L = params
        return min(np.sqrt(np.log((2*p+1)**2)/(slope*N+intercept)) + L,1)
    elif len(params) == 5:
        slope,intercept,L,k,x0 = params
        return min(np.sqrt(np.log((2*p+1)**2)/(slope*N+intercept)) + L/(1+np.exp(-k*(N-x0))),1)
    
def fit_mu_model(data,init):
    if len(init) == 2:
        L = np.mean(data[data['r']==1]['approxUB'].values)
        def model(N_log1r,slope,intercept):
            N, log1r = N_log1r.T
            return np.sqrt(log1r/(slope*N+intercept)) + L
    elif len(init) == 3:
        def model(N_log1r,slope,intercept,L):
            N, log1r = N_log1r.T
            return np.sqrt(log1r/(slope*N+intercept)) + L
    elif len(init) == 5:
        def model(N_log1r,slope,intercept,L,k,x0):
            N, log1r = N_log1r.T
            return np.sqrt(log1r/(slope*N+intercept)) + L/(1+np.exp(-k*(N-x0)))
        
    # Extract data for N, log1r, approx
    N_data = data['N'].values
    log1r_data = data['log(1/r)'].values
    approx_data = data['approxUB'].values
    N_log1r_data = np.column_stack((N_data, log1r_data))
    
    # Initial guesses for the parameters
    initial_guesses = init
    
    # Curve fitting
    params, covariance = curve_fit(model, N_log1r_data, approx_data, p0=initial_guesses,maxfev=2000)

    # Calculate predictions and mean squared error
    predicted_approx = model(N_log1r_data, *params)
    mse = mean_squared_error(approx_data, predicted_approx)
    
    if len(init) == 2:
        params = params.tolist() + [np.mean(data[data['r']==1]['approxUB'].values)]
        return params,mse
    else:
        return params,mse