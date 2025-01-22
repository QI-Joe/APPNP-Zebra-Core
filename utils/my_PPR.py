"""
Author: Qi Shihao

the code is write to implment approximate-PPR algorithm follow the strcuture of FORA
the paper detail can be found here https://gsai.ruc.edu.cn/uploads/20210522/80132ff9fdea401a97d42b6823ca1d5a.pdf
with inspire from link: https://wyydsb.xin/DataMining/ppr.html and https://blog.csdn.net/weixin_48167662/article/details/118993411
"""

from collections import defaultdict
import copy
import heapq
from logging import warning
import random
import time
from aiohttp import UnixConnector
from click import style
from flask import config
from torch import Tensor
import numpy as np
from abc import ABC, abstractmethod
import queue
import torch_geometric
from torch_geometric.data import Data
import multiprocessing
import sys

import torch_geometric.utils

sys.path.append("/mnt/d/CodingArea/Python/TestProejct")
from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting, Dynamic_Dataloader, to_cuda
from Testing_code.uselessCode import create_propagator_matrix, ppr_matrix_comparsion, Approx_personalized_pagerank, \
    Approx_personalized_pagerank_torch, ppr_comparsion, tppr2matrix, TPPR_Simple, tppr_querying
import scipy.sparse as sprs
import torch
from typing import Union
import tracemalloc
import linecache
import os
from functools import partial
from numba import jit

from concurrent.futures.process import ProcessPoolExecutor

class Storage:

    class key_class:
        def __init__(self, key: tuple[int, int]) -> None:
            self.key = key
            self.value_ptr: Storage.value_class = None
        
        def get_value(self):
            if self.value_ptr == None:
                warning("the pointer point to Null value")
            return self.value_ptr
        
        def match_value(self, value: object):
            self.value_ptr = value

        def return_key(self):
            return self.key
    
    class value_class:
        def __init__(self, value: tuple) -> None:
            self.value = value
            self.key_ptr:Storage.key_class = None

        def match_key(self, key: object):
            self.key_ptr = key

        def get_key(self):
            return self.key_ptr
        
        def return_value(self):
            return self.value

    def __init__(self, key: tuple, value: float):
        self.key = self.key_class(key) # Storage.key_class
        self.value = self.value_class(value) # Storage.value_class
        self.key.match_value(self.value)
        self.value.match_key(self.key)
    
    def __lt__(self, new_storage_: object):
        return self.value.value > new_storage_.value.value
    
    def __add__(self, new_others: object):
        return Storage((-1,-1), (self.value.value + new_others.value.value))
    
    def get_key(self):
        return self.key
    
    def get_key_value(self):
        return self.key.key


class ModifiedHeapq:
    def __init__(self) -> None:
        self.pq: list[Storage] = []
        self.key_index: dict[tuple[int], Storage.key_class] = dict()

    def push(self, element: Storage):
        heapq.heappush(self.pq, element)
        key_value, key_ptr = element.get_key_value(), element.get_key()
        self.key_index[key_value] = key_ptr

    def pop(self)->Storage:
        if self.pq != None:
            storage_: Storage = heapq.heappop(self.pq)
            key = storage_.key.return_key()
            del self.key_index[key]
            return 
        return None
            
    def __getitem__(self, index:int):
        """
        expect only get index==0, and utility is equlivient to peek()
        means always return the first elements
        """
        return self.pq[index]
    
    def __len__(self):
        return self.pq.__len__()
    
    def _is_empty(self):
        return len(self.pq) <= 0

    def has_key(self, outer_key: tuple[int, int]):
        return outer_key in self.key_index.keys()

    def get_value(self, outer_key):
        return self.key_index[outer_key].value_ptr.value

    def modify(self, key: tuple[int,int], value: float):
        key_object = self.key_index[key]
        value_ptr: Storage.value_class = key_object.get_value()
        value_ptr.value = value
        heapq.heapify(self.pq)

        

class FORA(ABC, object):
    def __init__(self, edge_idx: Tensor, nodes: Tensor, alpha: float, \
                 epsilon: float, failure_rate: float, delta: float, k: int = 100, r_max: float = None):
        super(FORA, self).__init__()
        self.edges = edge_idx
        self.nodes = nodes
        self.alpha = alpha
        self.r_max = r_max
        self.confirm_extra_(epsilon, failure_rate, delta)
        self.n_num_nodes = nodes.shape[0]
        self.m_num_edges = edge_idx.shape[0]
        self.k = k

        if self.r_max == None:
            self.r_max = self.compute_r_max()
        self.neighbor_out = self.sort_neighbor(self.edges, self.nodes)
        self.zero_upper_bound_ppr: float = 1.0

        self.residual_cache_ = None
        self.ppr_cahce_ = None
    
    @abstractmethod
    def compute_r_max(self):
        pass

    def sort_neighbor(self, edges: Tensor, nodes: list):
        node_dict: dict[int, list[int]] = dict()
        edges = edges.T.numpy()

        for singledge in edges:
            src, dist = singledge
            if src not in node_dict.keys():
                node_dict[src] = [dist]
            else:
                node_dict[src].append(dist)
        return node_dict

    def confirm_extra_(self, epsilon, failure_rate, delta):
        self.epsilon = epsilon
        self.failure_rate = failure_rate
        self.delta = delta

    def compute_r_max(self):
        self.r_max = (self.epsilon / self.m_num_edges**(1/2)) * (self.delta / ((2*self.epsilon/3+2)*np.log(2/self.failure_rate)))**(1/2)
        return self.r_max

    def construct_(self, node_list: list, target: int, construct_type: bool= False):
        """
        used to build a 2-hop neighbor finder defaultdict, return a defaultdict
        
        :param construct_type is used to dictinct ppr dict and residual dict; true for ppr, false for residual
        when ppr build no quick index (key list) will be returned, but []
        """
        _dict = defaultdict(float)
        quick_idx = []
        for node in node_list:
            if node ==target and not construct_type:
                _dict[(target, node)] = 1
            else:
                _dict[(target, node)] = 0
            
            if not construct_type: quick_idx.append((target, node))
        return _dict, quick_idx

    def update_residual_dict(update: defaultdict):
        ...
    
    def update_ppr_dict(update: defaultdict):
        ...
    
    def node_degree(self, single_node: int):
        if single_node not in self.neighbor_out.keys():
            return None
        return len(self.neighbor_out[single_node])

    def get_neighbor(self, single_node: int):
        if single_node not in self.neighbor_out.keys():
            return None
        return self.neighbor_out[single_node]

    def kth_element(self, ppr_: defaultdict):
        return sorted(ppr_.items(), key=lambda x: x[1])[self.k if self.k<len(ppr_) else -1]

    def serilize_parameter(self):
        min_delta = 1.0 / self.n_num_nodes
        if self.k==None or self.k == 0: self.k=500
        init_delta = 1/(self.k*10 if self.n_num_nodes > (self.k*10) else self.n_num_nodes/2)

        new_pfail = 1.0/self.n_num_nodes / self.n_num_nodes / np.log(self.n_num_nodes)
        self.failure_rate = new_pfail
        self.delta = init_delta
        return min_delta, new_pfail, init_delta

    def fora_query_multithreading(self, out_side_query_list: Union[list, Tensor, np.ndarray], used_process: int = 10):
        """
        assume known parameter included in class
        :param query_list, a batch of node idx
        :param edge_index
        """
        # prepare the parameters
        if isinstance(out_side_query_list, Tensor):
            out_side_query_list = out_side_query_list.cpu().numpy()

        # self.time_queue = queue.Queue()
        # self.lock = multiprocessing.Lock()

        min_delta, new_pfail, delta = self.serilize_parameter()
        # build up the processing pool
        ppr_list: dict[int, defaultdict] = {}
        with ProcessPoolExecutor(max_workers=used_process) as executor:
            # partial function can used for pass parameter with name
            worker = partial(self.fora_query_single_point_, delta=delta, min_delta=min_delta, \
                             new_pfail=new_pfail, multiwork=True, mode="linear")
            futures = [executor.submit(worker, node_idx) for node_idx in out_side_query_list]
            for exe in futures:
                residual_, ppr, src = exe.result(60)
                ppr_list[src] = ppr
        return dict(sorted(ppr_list.items(), key=lambda item: item[0]))


    def fora_query_single_point_(self, target: int, delta: float=None, min_delta: float=None, new_pfail: float=None, \
                                 multiwork: bool=False, mode: str = "heap"):
        if not multiwork:
            min_delta, new_pfail, delta = self.serilize_parameter()
        
        lowest_delta_rmax = self.epsilon*np.sqrt(min_delta / (3*self.m_num_edges*np.log(2/new_pfail)))

        # lock = multiprocessing.Lock()

        r_sum = 1.0
        forward_list: list = [target]

        mode = mode.lower()
        iteration = 0
        while (delta>=min_delta):
            rmax = self.compute_r_max()
            iteration+=1

            fwd_time = time.time()
            if mode == "linear":
                forward_list, r_sum, residual_, ppr_ = self.forward_push_linear(target=target, rsum=r_sum, rmax = rmax,
                                                    lowerst_rmax=lowest_delta_rmax, forward_list=forward_list)
            else: 
                residual_, ppr_ = self.forward_push_heap(target=target)
            fwd_time = time.time() - fwd_time
            mc_time = time.time()
            ppr_theta, _ = self.MC_perfection(target=target, iteration=iteration, r_sum=r_sum, residual=residual_, ppr_dict=ppr_)
            mc_time = time.time() - mc_time
            kth_ppr_score = self.kth_element(ppr_theta)[1]
            
            # lock.acquire()
            # self.time_queue.put((fwd_time, mc_time, target))
            # lock.release()
            # print(f"min threshold: {round(min_delta, 6)}, current delta mintor: {round(delta, 6)}, \
            #       kth value is: {self.k} and value is {kth_ppr_score}")
            if (kth_ppr_score>=(1+self.epsilon)*delta or delta<=min_delta):
                self.delta = delta
                break
            else:
                delta = max(min_delta, delta/4.0)
            self.delta = delta
        

        print(f"#--------------------------------------------\nnode {target} has been compute finished")
        print(f"src node on {target} fwd_time cost {round(fwd_time, 4)}s, mente corlo perfection cost {round(mc_time, 4)}s")
        if multiwork:
            return residual_, ppr_theta, target

        return residual_, ppr_theta

    def storage2dict(self, residuals: list[Storage]):
        back_ = dict()
        for item in residuals:
            key, value = item.key.key, item.value.value
            back_[key] = value
        return back_

    def forward_push_linear(self, target, rsum: float, rmax: float, lowerst_rmax: float, forward_list: list[int]):
        """
        :param forward_list, is a list contain initial src point as storage
        :param target, src node
        """
        epsilon: float = rmax

        in_forward: list[bool] = [False]*self.n_num_nodes # this s, v round node
        in_next_forward: list[bool] = [False] * self.n_num_nodes # next s, v round node

        next_forward_from: list[int] = list()

        init_vs_pattern = (target, forward_list[0])
        residual_ = defaultdict(float, {init_vs_pattern: 1.0})
        ppr_ = defaultdict(float)

        for nodeid in forward_list: # forward list always stores qualified v not u
            in_forward[nodeid] = True
        
        i=0
        while i < len(forward_list):
            v = forward_list[i]
            i+=1
            in_forward[v] = False

            vs_pattern = (target, v)

            out_degree = self.node_degree(v) if self.node_degree(v)!=None else 1
            if residual_[vs_pattern]/out_degree >= epsilon:
                v_residue = residual_[vs_pattern]
                residual_[vs_pattern] = 0
                if vs_pattern in ppr_.keys():
                    ppr_[vs_pattern] += v_residue*self.alpha
                else:
                    ppr_[vs_pattern] = v_residue*self.alpha
                
                rsum -= v_residue*self.alpha

                if out_degree<=1: # in fact we assume every node has a self-loop, otherwise should be 0
                    """
                    regrad this as a sitatuion where s->v and v is a isolated point, so in v->u propagation u could only be s
                    thus, the s->u could only become s->s; but attention, this explination may not work since grpah in paper
                    is direct graph not in-direct graph                    
                    """
                    self_loop_vs_pattern = (target, target) # or (v, target) ?
                    residual_[self_loop_vs_pattern] += v_residue * (1-self.alpha)

                    target_degree = self.node_degree(target) if self.node_degree(target) != None else 1
                    if (target_degree>0+1 and in_forward[target] != True and residual_[self_loop_vs_pattern] / target_degree >= epsilon):
                        forward_list.append(target)
                        in_forward[target] = True
                    elif (target_degree>=0+1 and in_next_forward[target] != True and residual_[self_loop_vs_pattern]/target_degree>=lowerst_rmax):
                        next_forward_from.append(target)
                        in_next_forward[target] = True
                    
                    continue

                avg_push_residual = ((1-self.alpha) * v_residue) / out_degree
                for u in self.get_neighbor(v):
                    su_pattern = (target, u)
                    if (su_pattern not in residual_.keys()):
                        residual_[su_pattern] = avg_push_residual
                    else:
                        residual_[su_pattern] += avg_push_residual
                    
                    u_degree = self.node_degree(u)
                    if (u_degree!=None and in_forward[u]!=True and residual_[su_pattern]/u_degree >= epsilon):
                        forward_list.append(u)
                        in_forward[u] = True
                    elif (u_degree!=None and in_next_forward[u]!=True and residual_[su_pattern]/u_degree>=epsilon):
                        next_forward_from.append(u)
                        in_next_forward[u]=True

        forward_list = next_forward_from
        return forward_list, rsum, residual_, ppr_


    def forward_push_heap(self, target: int, node_list: list[int] = None)-> tuple[ModifiedHeapq, defaultdict]:
        # residual_dict = defaultdict(float)
        ppr_dict = defaultdict(float)
        init_pattern = Storage(key=(target, target), value=1)

        heap: ModifiedHeapq = ModifiedHeapq()
        heap.push(init_pattern)
        max_iteration = np.ceil(1/self.r_max)
        if max_iteration > 1000: max_iteration = 200

        while not heap._is_empty() and max_iteration:
            top_vs_pattern: Storage = heap[0]
            vs_residual, vs_pattern = top_vs_pattern.value.value, top_vs_pattern.key.key
            src, dist = vs_pattern

            # PPR update
            node_degree = self.node_degree(dist)
            if node_degree==None: 
                node_degree = 1
            if vs_residual / node_degree < self.r_max:
                break # why break? becuase heap always keep the largest value at top
            heap.pop() # equals to r(s,v) = 0

            if vs_pattern in ppr_dict.keys():
                ppr_dict[vs_pattern] += self.alpha * vs_residual
            else:
                ppr_dict[vs_pattern] = self.alpha * vs_residual

            # find and build first neighbor of target
            neighbors = self.get_neighbor(dist)
            if neighbors==None:
                continue

            v_degree = node_degree
            neighbor_list, quick_idx = self.construct_(neighbors, src, construct_type=False)
            residual_s = (1-self.alpha)*(vs_residual/v_degree)
            for key_u in quick_idx:
                if heap.has_key(key_u):
                    heap.modify(key_u, heap.get_value(key_u) + residual_s)
                else:
                    heap.push(Storage(key_u, residual_s))


            max_iteration -= 1
            # print(f"the pattern on nodes {vs_pattern} on iteration {max_iteration}")

        # print()
        return heap, ppr_dict

    def drand(self):
        np.random.seed(int(time.time()))
        return np.random.rand()
    
    def bernoulling_drand(self):
        np.random.seed(int(time.time()))
        rng = np.random.default_rng()
        return rng.random() < self.alpha

    def random_walk(self, src: int, dist: int, alpha: float, max_step: int = 10):
        walk_queue = queue.Queue()
        walk_queue.put((src, dist))
        if max_step <= 0: max_step_copy=10
        else: max_step_copy = max_step
        src_s, next_dist_w = -1, src

        while not walk_queue.empty() and max_step>0:
            src_s, dist_v = walk_queue.get()
            assert src_s == src, f"source node has been modfied, pls check code logic. Expected {src}, get {src_s}"

            neighbors_v = self.get_neighbor(dist_v)
            if neighbors_v == None: continue

            next_dist_w = random.sample(neighbors_v, k=1)[0]
            if self.drand() < self.alpha or self.bernoulling_drand():
                return src_s, next_dist_w, abs(max_step - max_step_copy)
            else: 
                walk_queue.put((src_s, next_dist_w))
            max_step-=1
        return src_s, next_dist_w, max_step

    def MC_perfection(self, target, iteration: int, r_sum: float, residual: Union[ModifiedHeapq | defaultdict], ppr_dict: defaultdict):
        residual_ = copy.deepcopy(residual) # if its heap, then need use residual_[idx].value.value to access
        if isinstance(residual_, ModifiedHeapq):
            residual_ = self.storage2dict(residual_.pq)
            r_sum = sum(residual.pq, start=Storage(key=(0,0), value=0.0)).value.value
        
        if len(residual_)==0 or r_sum == 0:
            return ppr_dict, ppr_dict

        omega = r_sum * ((2*self.epsilon/3+2)*np.log(2/self.failure_rate)) / (self.epsilon**2*self.delta)
        self.omega = copy.deepcopy(omega)

        pi_theta = copy.deepcopy(ppr_dict)

        for key, val in residual_.items():
            if val == 0.0: continue
            omega_v = int(np.ceil(val * omega / r_sum))
            alpha_v = val * omega / (r_sum*omega_v)

            if omega_v>1000: omega_v=200

            for epoch in range(omega_v):
                src, final_dist_t, walk_step = self.random_walk(src=target, dist=target, alpha=alpha_v)
                if (src, final_dist_t) in pi_theta.keys():
                    pi_theta[(src, final_dist_t)] += alpha_v*r_sum / omega
                else:
                    pi_theta[(src, final_dist_t)] = alpha_v*r_sum / omega
                
            # print(f"at event {key} with total {omega_v}, take random walk for {walk_step} steps")
        # print(f"src node {target}, with {iteration}th iteration residual list with length {residual_.__len__()}")
        return pi_theta, ppr_dict
    
    def calculate_lambda(self, r_sum: float, pfail: float, upper_bound: float, total_omega: float):
        # attention, in paper it should start with 2.0/3 ...
        return 1.0/3*np.log(2/pfail) * r_sum + \
        np.sqrt(4.0/9.0*np.log(2.0/pfail)*r_sum*r_sum+\
                8*total_omega*np.log(2.0/pfail)*r_sum*upper_bound) \
                /(2.0*total_omega)

    def topk_bound_check(self, r_sum: float, total_omega: float):
        pfail: float = self.failure_rate / (self.n_num_nodes * np.log(self.n_num_nodes))

        min_ppr = 1.0/self.n_num_nodes
        sqrt_min_ppr = np.sqrt(1.0/self.n_num_nodes)

        epsilon_v = np.sqrt(2.67*r_sum*np.log(2.0/pfail)/total_omega)
        self.zero_upper_bound_ppr = self.calculate_lambda(r_sum=r_sum, pfail=pfail, \
                                                          upper_bound=self.zero_upper_bound_ppr, total_omega=total_omega)

        for key in self.ppr_list.keys():
            ...
        ...

    def exact_ppr1(self):
        """
        basically a independent part of FORA built for test. Write inside matrix is becuase it's easy to call other function
        test dataset will be part of Cora Dataset
        """
        # convert given edge_idx to normalized adjacency matrix
        # sparse_norm_adj = PyG_to_Adj(self.edges)
        propagator_matrix = create_propagator_matrix(graph=self.edges, alpha=0.1)
        return propagator_matrix

    def approx_ppr_multiprocessing(self, dense_adj: Tensor, query_list: Union[list, np.ndarray]):
        qsize = query_list.shape[0]
        interval = query_list.shape[0] // 10
        
        queries = [query_list[start: start+interval] for start in range(0, qsize-1, interval)]
        id_tuple = {}
        with ProcessPoolExecutor(max_workers=10) as executor:
            worker = partial(self.sub_linear, dense_adj = dense_adj)
            exe = [executor.submit(worker, query, idx) for idx, query in enumerate(queries)]

            for ele in exe:
                result, idx = ele.result()
                id_tuple[idx] = result
            id_tuple = dict(sorted(id_tuple.items(), key=lambda item: item[0]))
        return np.vstack(tuple(id_tuple.values()))

    def sub_linear(self, query_list, processing_id: int, dense_adj: Tensor):
        print(f"branch {processing_id} get started")
        numpy_matrix = np.array([])
        for index in query_list:
            idx_query = Approx_personalized_pagerank(dense_adj, index, alpha=self.alpha)
            numpy_matrix = np.vstack((numpy_matrix, idx_query)) if len(numpy_matrix) else idx_query
        print(f"branch {processing_id} get finished")
        return numpy_matrix, processing_id

    def approx_ppr(self, edge_idx: Tensor, query_list: Union[list, np.ndarray]):
         
        dense_adj = torch_geometric.utils.to_dense_adj(edge_index=edge_idx)
        if isinstance(query_list, int):
            return Approx_personalized_pagerank_torch(dense_adj, query_list, alpha=self.alpha)
        
        n = len(query_list)
        _matrix: np.ndarray = np.array([])
        for index in query_list:
            print(f"node {index} start query") if (index+1) % 50 == 0 else None
            idx_query = Approx_personalized_pagerank_torch(dense_adj, index, alpha=self.alpha)
            if isinstance(idx_query, Tensor):
                _matrix = torch.vstack((_matrix, idx_query)) if len(_matrix) else idx_query
            else:
                _matrix = np.vstack((_matrix, idx_query)) if len(_matrix) else idx_query
            print(f"node {index} end query") if (index+1) % 50 == 0 else None

        return _matrix

    def ppr2_csr_convertion(self, num_nodes: int, edge_idx: Tensor=None)->sprs.coo_matrix:
        if edge_idx == None: edge_idx = self.edges
        row = edge_idx[0].numpy()
        col = edge_idx[1].numpy()

        assert len(row) == len(col), "row, col length is nto the same."
        data= np.ones(len(row))
        coo_matrix = sprs.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        csr_matrix = coo_matrix.tocsr()
        return csr_matrix

    def ppr(self, edge_idx, num_nodes, alpha=0.1, personalize=None, reverse=False):
        """ Calculates PageRank given a csr graph

        Inputs:
        -------

        G: a csr graph.
        p: damping factor
        personlize: if not None, should be an array with the size of the nodes
                    containing probability distributions.
                    It will be normalized automatically
        reverse: If true, returns the reversed-PageRank

        outputs
        -------

        PageRank Scores for the nodes

        """
        # In Moler's algorithm, $A_{ij}$ represents the existences of an edge
        # from node $j$ to $i$, while we have assumed the opposite!
        p = 1-alpha
        A = self.ppr2_csr_convertion(edge_idx=edge_idx, num_nodes = num_nodes)
        if reverse:
            A = A.T

        n, _ = A.shape
        r = np.asarray(A.sum(axis=1)).reshape(-1)

        k = r.nonzero()[0]

        D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

        if personalize is None:
            personalize = np.ones(n)
        personalize = personalize.reshape(n, 1)
        s = (personalize / personalize.sum()) * n

        I = sprs.eye(n)
        x = sprs.linalg.spsolve((I - p * A.T @ D_1), s)

        x = x / x.sum()
        return x

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def default_setting():
    return {
        "alpha": 0.15,
        "epsilon": 0.05,
        "failure_rate": 0.05,
        "delta": 0.05,
    }

def cora_data(batch_size = 10) -> tuple[Data, int]:
    from my_dataloader import data_load
    from torch_geometric.loader import NeighborLoader
    cora_test_set, idxloader = data_load("cora")
    neighborloader = NeighborLoader(data = cora_test_set, batch_size=batch_size, num_neighbors=[-1], shuffle=False, input_nodes=torch.arange(1800, 1900))
    cora1: Data = next(iter(neighborloader))
    cora1_num_nodes = cora1.num_nodes
    return cora1, cora1_num_nodes

def normalize_matrix(matrix: Tensor):
    """
    fuck off, just for a better comparsion result
    """
    sum_vector = matrix.sum(dim=1, keepdim=True)
    return matrix/sum_vector

def defaultdict2matrix(ppr_list: list[defaultdict], num_nodes: int = 10):
    values = []
    row_indices = []
    col_indices = []

    for ele in ppr_list:
        for node, value in ele.items():
            # Assuming node is the target node for the edge
            row_indices.append(node[0])  # The row index where the value is located
            col_indices.append(node[1])  # The index of the ppr_list (source node)
            values.append(value)

    # Convert to tensors
    row_indices = torch.tensor(row_indices, dtype=torch.int64)
    col_indices = torch.tensor(col_indices, dtype=torch.int64)
    values = torch.tensor(values, dtype=torch.float32)

    # Create sparse matrix
    indices = torch.vstack((row_indices, col_indices))
    sparse_matrix = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=torch.Size([num_nodes, num_nodes])
    )

    return sparse_matrix.to_dense()
    

def manual_data() -> tuple[Data, int]:
    import torch
    test_storage_set = [
        [0,1], [0,5], [1,3], [1,4], [1,8], [2,4], [2,8], [2,6], [3,0], [4,3], [5,2], [5,6], [6,7], [7,2], [9,6]
    ]
    test_storage_set = torch.tensor(test_storage_set).T
    test_manual = Data(x = torch.arange(10), edge_index = test_storage_set)
    test_num_nodes = test_manual.num_nodes
    return test_manual, test_num_nodes

def log_queue_data(queue: queue.Queue):
    # Retrieve all items from the queue
    items = []
    
    while not queue.empty():
        items.append(queue.get())

    # Sort items by target
    items.sort(key=lambda x: x[2])  # Sort by the third element (target)

    # Write sorted items to a text file
    with open('logs/timelog/queue_log.txt', 'w') as f:
        for time1, time2, target in items:
            f.write(f'Time1: {round(time1, 4):.4f}, Time2: {round(time2, 4):.4f}, Target: {target}\n')

def random_loading(dataset: str):
    data, idxloader = data_load(dataset)

    return data, data.num_nodes

if __name__ == "__main__":
    dt = "mathoverflow"
    test_data, test_num_nodes = random_loading(dt)
    graph_list = Temporal_Splitting(test_data, 3).temporal_splitting()
    dataneighbor = Dynamic_Dataloader(graph_list, graph=test_data)
    test_data = dataneighbor.get_temporal()
    test_data = to_cuda(test_data, "cpu")
    print(test_data.x.shape, test_data.edge_index.shape)

    extra_parameter = default_setting()
    fora = FORA(edge_idx=test_data.edge_index, nodes = test_data.x, **extra_parameter)
    query_list = np.arange(test_data.x.size(0))
    
    
    tracemalloc.start()
    time1 = time.time()

    # matrix_approx_ppr = fora.approx_ppr(edge_idx=test_data.edge_index, query_list=query_list)
    # matrix_approx_ppr = torch.load(r"logs/temporary_running_result/Matrix_approx_cora.pt")
    # torch.save(matrix_approx_ppr, r"logs/temporary_running_result/fora_mathoverflow.pt")

    matrix_ppr_time = time.time() - time1
    matrix_approx_snapshot = tracemalloc.take_snapshot()

    time2 = time.time()

    """TPPR Computation"""
    alpha_list, beta_list = [0.1, 0.1], [0.05, 0.95]
    topk, node_num = 10, test_data.x.shape[0]+10
    tppr = TPPR_Simple(alpha_list=alpha_list, node_num=node_num, beta_list=beta_list, topk=topk)
    abs_time, (tppr_node, tppr_weight) = tppr_querying(dataset=dt, tppr=tppr, input_dt=test_data)
    fora_time = time.time()-abs_time
    _, tppr_matrix = tppr2matrix(tppr_node=tppr_node, tppr_weight=tppr_weight)
    approx_ppr_matrix = tppr_matrix.to_dense()

    """
    Fora multi-processing and for loop computation
    """
    # ppr_list = []
    # for querys in query_list:
    #     _, ppr_dict = fora.fora_query_single_point_(target=querys, mode="heap")
    #     ppr_list.append(ppr_dict)
    # ppr_dict = fora.fora_query_multithreading(query_list, 10)
    # ppr_list = list(ppr_dict.values())
    # approx_ppr_matrix = defaultdict2matrix(ppr_list, test_num_nodes)
    
    fora_snapshot = tracemalloc.take_snapshot()
    # fora_time = time.time() - time2
    print(f"running time with {round(fora_time, 4)}")
    

    # matrix_approx_ppr = torch.load(r"logs/temporary_running_result/Matrix_approx_cora.pt")
    # approx_ppr_matrix = torch.load(r"logs/temporary_running_result/fora_cora.pt")

    # approx_ppr_matrix, matrix_computation1 = normalize_matrix(approx_ppr_matrix), normalize_matrix(matrix_computation1)
    # qualified_rows, matched, weight_diff, selected_weight_diff, full_eq = ppr_matrix_comparsion(ppr1=approx_ppr_matrix, ppr2=matrix_approx_ppr)
    
    # print(f"Total number length {test_num_nodes}")
    # print(f"Qualified Rows: {qualified_rows.shape}")
    # print(f"Matched: {matched.shape}")
    # print(f"Weight Difference: {weight_diff}, avergage difference {round(torch.sum(weight_diff).item()/test_num_nodes, 4)}\n")

    # print("Exact PPR Simluation Memory Cost")
    # display_top(matrix_snapshot)
    print("\nApproximate PPR Simulation Memory Cost")
    display_top(matrix_approx_snapshot)
    print("\nFora Approxmiate PPR Memory Cost")
    display_top(fora_snapshot)

    print("Approx matrix time is:", round(matrix_ppr_time, 5))
    print("fora time is:", round(fora_time, 5))