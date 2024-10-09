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
    return results


def read_data(N_list, idx_list, get_path, metric='approxUB', opt ='max', num_processes=4, resolution = 800):
    log_inv_r_array = np.linspace(0, 8.8, resolution)
    if resolution == 1:
        log_inv_r_array = np.array([0])
    r_array = 1 / np.exp(log_inv_r_array)
    
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


def fitted_mu(N,p,params):
    if len(params) == 3:
        slope,intercept,L = params
        # return min(np.sqrt(np.log((2*p+1)**2)/(slope*N+intercept)) + L,1)
        return np.sqrt(np.log((2*p+1)**2)/(slope*N+intercept)) + L
    elif len(params) == 5:
        slope,intercept,L,k,x0 = params
        # return min(np.sqrt(np.log((2*p+1)**2)/(slope*N+intercept)) + L/(1+np.exp(-k*(N-x0))),1)
        return np.sqrt(np.log((2*p+1)**2)/(slope*N+intercept)) + L/(1+np.exp(-k*(N-x0)))
    
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