from gap_approx_opt import sample_gap_approx_opt
from obj import distribution_k_colorable_graph
import multiprocessing

import json,argparse,os
import numpy as np

PROBLEM_PATH = "./n9_18_maxKcolorable.json"
DISTRIBUTION_FOLDER = f'./distribution/max_k_colorable'
RESULT_FOLDER = f'./approx_result/max_k_colorable'    
TASK = 'approx'

def get_problem(n,k,idx):
    # get problem graph
    with open(PROBLEM_PATH, "r") as json_file:
        problem_dict = json.load(json_file)
    edgelist = problem_dict[str(n)][str(k)][int(idx)]
    
    problem_dict = {"edges": edgelist, "num_nodes": int(n), "num_colors": int(k)}
    
    return problem_dict

def get_distribtuion_path(n,k,idx):
    # get distribution of obj values
    distribution_folder = f'{DISTRIBUTION_FOLDER}/n{n}_k{k}'
    if not os.path.exists(distribution_folder):
        os.makedirs(distribution_folder)
        
    dist_path = f'{distribution_folder}/{idx}.npy'
    return dist_path

def get_result_path(n,k,idx,p):
    result_folder = f'{RESULT_FOLDER}/n{n}_k{k}/p{p}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    save_path = f'{result_folder}/{idx}.json'
    return save_path


def worker(params):
    sample_gap_approx_opt(params,TASK,get_problem,get_distribtuion_path,distribution_k_colorable_graph,get_result_path)
    

def main():
    parser = argparse.ArgumentParser(description='run for max_k_colorable_graph problems')
    parser.add_argument('--N', type=str, default='11,12,13,14,15,16,17', help='nodes')
    parser.add_argument('--K', type=int, default=3, help='colors')
    parser.add_argument('--indices', type=str, default='0_47', help='colors')
    parser.add_argument('--levels', type=str, default='0,1,3,5,7,9', help='QAOA levels (round, depth, layer number)')
    
    args = parser.parse_args()
    level_list = [int(p) for p in args.levels.split(',')]
    
    print(f"N:{args.N},K:{args.K},indices:{args.indices},levels:{args.levels}")
    
    N_list = [int(n) for n in args.N.split(',')]
    K = args.K
    indices_range = [int(idx) for idx in args.indices.split('_')]
        
    for p in level_list:
        print(f"depth:{p}")
        args_list = [[n, K, idx, p] for n in N_list for idx in range(indices_range[0],indices_range[1]+1)]
        
        num_processors = min([48,multiprocessing.cpu_count()])
        
             # 设置multiprocessing启动方法为'spawn'
        multiprocessing.set_start_method('spawn', force=True)

        with multiprocessing.Pool(processes=num_processors) as pool:
            # Map the worker function over the arguments list
            pool.map(worker, args_list)
        
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        

if __name__ == '__main__':
    main()
