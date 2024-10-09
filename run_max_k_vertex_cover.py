from depth_progress import depth_progress_opt
from obj import distribution_max_k_vertex_cover
import multiprocessing

import json,argparse,os
import numpy as np

def get_problem(n, idx, PROBLEM_PATH):
    # get problem graph
    with open(PROBLEM_PATH, "r") as json_file:
        problem_dict = json.load(json_file)
    edgelist = problem_dict[str(n)][int(idx)]
    
    problem_dict = {"N":n,"edgelist": edgelist}
    
    return problem_dict

def get_distribution_path(n,idx,DISTRIBUTION_FOLDER):
    # get distribution of obj values
    distribution_folder = f'{DISTRIBUTION_FOLDER}/n{n}'
    if not os.path.exists(distribution_folder):
        os.makedirs(distribution_folder)
        
    dist_path = f'{distribution_folder}/{idx}.npy'
    return dist_path

def get_result_path(n,idx,p,result_folder):
    save_path = f'{result_folder}/n{n}/p{p}/{idx}.json'
    return save_path

def worker(params):
    n, idx, args, PROBLEM_PATH, DISTRIBUTION_FOLDER, result_folder, levels = params
    depth_progress_opt(
        [n,idx],
        args.task,
        lambda n, idx: get_problem(n, idx, PROBLEM_PATH),
        lambda n, idx: get_distribution_path(n, idx, DISTRIBUTION_FOLDER),
        distribution_max_k_vertex_cover,
        lambda n, idx, p: get_result_path(n, idx, p, result_folder),
        levels,
        args.th
    )

def main():
    parser = argparse.ArgumentParser(description='run for max k vertex cover')
    parser.add_argument('--N', type=str, default='18,20,22,24,26,28,30', help='nodes')
    parser.add_argument('--indices', type=str, default='0_47', help='colors')
    parser.add_argument('--task', type=str, default='approx')
    parser.add_argument('--th', type=bool, default=False, help='if use th')
    args = parser.parse_args()
    
    PROBLEM_PATH = "./18_30_max_k_vertex_cover.json"
    DISTRIBUTION_FOLDER = f'./distribution/max_k_vertex_cover'
    levels = [i for i in range(1,50)]
    if args.th:
        if args.task == 'approx':
            result_folder = f'./approx_result/th/max_k_vertex_cover'
        elif args.task == 'max':
            result_folder = f'./popt_result/th/max_k_vertex_cover'
    else:
        if args.task == 'approx':
            result_folder = f'./approx_result/max_k_vertex_cover'
        elif args.task == 'max':
            result_folder = f'./popt_result/max_k_vertex_cover'

    N_list = [int(n) for n in args.N.split(',')]

    for n in N_list:
        for p in levels:
            result_path = f'{result_folder}/n{n}/p{p}'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                
    indices_range = [int(idx) for idx in args.indices.split('_')]
    indices = [idx for idx in range(indices_range[0], indices_range[1] + 1)]
    
    print('levels:')
    print(levels)
    print('problemSize:')
    print(N_list)
    print('problemIndices:')
    print(indices)

    args_list = [[n, idx, args, PROBLEM_PATH, DISTRIBUTION_FOLDER, result_folder, levels] for n in N_list for idx in indices]
    num_processors = min([48, multiprocessing.cpu_count()])
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_processors) as pool:
        pool.map(worker, args_list)
    pool.close()
    pool.join()
        
if __name__ == '__main__':
    main()


