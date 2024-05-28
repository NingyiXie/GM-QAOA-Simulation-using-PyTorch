import numpy as np
from itertools import product, islice, permutations

from tqdm import tqdm

import os

def obj_k_colorable_graph(solution,edgelist):
    obj = len(edgelist)
    for i,j in edgelist:
        if solution[i] == solution[j]:
            obj -= 1
    return obj


def distribution_k_colorable_graph(edgelist,N,K):
    distribution = {}
        
    for solution in tqdm(product(list(range(K)), repeat=N),total=K**N):
        obj = obj_k_colorable_graph(solution,edgelist)
        distribution[obj] = distribution.get(obj,0) + 1
    
    distribution_array = np.array([[key, value] for key, value in distribution.items()],dtype=np.int64)
    
    return distribution_array


def obj_max_k_vertex_cover(edgelist, solution):
    # Convert the solution list to a set for faster lookup
    solution_set = set(solution)
    obj = 0
    
    # Iterate through each edge in the edge list
    for edge in edgelist:
        i,j,k =edge.split(" ")
        i = int(i)
        j = int(j)
        # Increment obj if either vertex of the edge is in the solution set
        if j in solution_set or i in solution_set:
            obj += 1
    return obj
    
def distribution_max_k_vertex_cover(N,edgelist):
    k=int(N/2)
    dis={}
    sequence=range(N)

    for combo in tqdm(combinations(sequence, k),total = math.comb(N, k)):
        obj=obj_max_k_vertex_cover(edgelist,combo)
        dis[obj]=dis.get(obj,0)+1
        
    distribution_array = np.array([[key, value] for key, value in dis.items()],dtype=np.int64)
    return distribution_array


def get_distances(locs):
    N = len(locs)
    distances = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j:
                x1,y1 = locs[i]
                x2,y2 = locs[j]
                distances[i,j] = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return distances


def obj_cvrp(permutation, depot_visit_decisions, capacity, items, distances):
    customer_num = len(items)

    current_customer = permutation[0]
    load = items[current_customer]
    obj = distances[-1, current_customer]

    for i in range(customer_num - 1):
        next_customer = permutation[i + 1]
        if load + items[next_customer] <= capacity and depot_visit_decisions[i] == 0:
            obj += distances[current_customer, next_customer]
            load += items[next_customer]
        else:
            obj += distances[current_customer,-1] + distances[-1, next_customer]
            load = items[next_customer]
        current_customer = next_customer

    obj += distances[current_customer, -1]
    return obj


def distribution_cvrp(coordinates,items,capacity):
    distances = get_distances(coordinates)
    
    customer_num = len(items)
    
    distribution = {}
    
    for perm in permutations(range(customer_num)):
        for dec in product([0, 1], repeat=customer_num - 1):
            obj = obj_cvrp(perm, dec, capacity, items, distances)
            distribution[obj] = distribution.get(obj,0) + 1
    
    distribution_array = np.array([[key, value] for key, value in distribution.items()],dtype=np.float32)
    
    return distribution_array



def obj_maxcut(solution, edge_array):
    # 将solution转换为numpy数组
    solution = np.array(solution)
    starts = edge_array[:, 0]
    ends = edge_array[:, 1]
    
    differences = solution[starts] - solution[ends]
    
    cut = np.sum(differences**2)
    
    return cut


def distribution_maxcut(edgelist,N):
    distribution = {}
    
    edge_array = np.array(edgelist)
    for solution in tqdm(product([0, 1], repeat=N)):
        obj = obj_maxcut(solution,edge_array)
        distribution[obj] = distribution.get(obj,0) + 1
    
    distribution_array = np.array([[key, value] for key, value in distribution.items()],dtype=np.int64)
    
    return distribution_array


def obj_tsp(permutation, distances):
    edges = np.array([[0] + permutation, permutation + [0]]).T
    obj = distances[edges[:, 0], edges[:, 1]].sum()
    return obj


def distribution_tsp(coordinates):
    n = len(coordinates)
    distances = get_distances(coordinates)
    distribution = {}
    for perm in tqdm(permutations(range(1,n))):
        obj = np.round(obj_tsp(list(perm), distances),6)
        distribution[obj] = distribution.get(obj,0) + 1
    
    distribution_array = np.array([[key, value] for key, value in distribution.items()],dtype=np.float32)
    
    return distribution_array