from scipy.io import mmread, mminfo
from scipy.sparse import csr_matrix, triu
import numpy as np
from tqdm import tqdm
from Converter import Converter
import time
import sys
import argparse

N_PARTITIONS = 64
isTranspose = 0
hdrf = 0

def read_inparts(filename):
    with open(filename, 'r') as f:
        return [int(line) for line in f]
    
def read_symmetric_adj(filename):
    start = time.time()
    info = mminfo(filename)
    adj = mmread(filename)
    adj = adj.tocsr()
    adj = adj.tocoo()
    #for i in zip(adj.row, adj.col):
        #print(i)
    #adj = adj.tocsr()
    print(f"Adj read completed! Time: {time.time() - start}")
    return adj, info

def create_edge_dict(adj, edge_part):
    #adj = adj.tocoo()
    #for i in zip(adj.row, adj.col):
        #print(i)
    start = time.time()
    edge_dict = {}
    edge_cntr = 0
    for edge in zip(adj.row, adj.col):
        u, v = edge
        edge_dict[f"{u} {v}"] = edge_part[edge_cntr]
        edge_cntr += 1
    print(f"Edge dict created! Time: {time.time() - start}")

    return edge_dict

def fill_bins(adj, 
              vtx_part,
              edge_dict):
    start = time.time()
    partitions = [[] for i in range(N_PARTITIONS)]
    for edge in zip(adj.row, adj.col, adj.data):
        u, v, data = edge
        partitions[edge_dict[f"{u} {v}"]].append((u, v, data))
    print(f"Partitioning completed! Time: {time.time() - start}")

    Wmax = -1
    accum = 0
    for part in partitions:
        if len(part) > Wmax:
            Wmax = len(part)
        accum += len(part)
    Wavg = accum / N_PARTITIONS
    print("Wmax: ", Wmax)
    print("Wavg: ", Wavg)
    print("Wmax/Wavg = ", Wmax/Wavg)
    print(len(partitions[0]))
    return partitions

def get_l2gMap(partition, 
               part_id,
               vtx_part): 
    locals = set()
    nonlocals_p1 = set()
    nonlocals_p2 = set()
    uniques = set()
    for edge in partition:
        u, v, data = edge
        if (vtx_part[u] == vtx_part[v]) and  (vtx_part[u] == part_id):
            if u not in uniques:
                uniques.add(u)
                locals.add((u, data))
        else:    
            if vtx_part[u] != part_id:
                if u not in uniques:
                    uniques.add(u)
                    nonlocals_p1.add((u, vtx_part[u], data))
            if vtx_part[v] != part_id:
                if v not in uniques:
                    uniques.add(v)
                    nonlocals_p2.add((v, vtx_part[v], data))
        
    locals = sorted(locals, key=lambda x: x[0])
    #print(locals)

    if hdrf:
        nonlocals = [*nonlocals_p1, *nonlocals_p2]
        nonlocals = sorted(nonlocals, key=lambda x: (x[1], x[0]))
        nonlocals = [(x[0], x[2]) for x in nonlocals]
    else:
        nonlocals_p1 = sorted(nonlocals_p1, key=lambda x: (x[1], x[0]))
        nonlocals_p2 = sorted(nonlocals_p2, key=lambda x: (x[1], x[0]))
        nonlocals_p1 = [(x[0], x[2]) for x in nonlocals_p1]
        nonlocals_p2 = [(x[0], x[2]) for x in nonlocals_p2]
    #print(nonlocals)
    #print("Locals: ", locals)

        if isTranspose == 1:
            nonlocals = nonlocals_p2
        else:
            nonlocals = nonlocals_p1
    return locals, nonlocals 

def get_local_csr(partition,
                  part_id,
                  locals,
                  nonlocals,
                  l2g_map,                  
                  little_m,
                  local_m,
                  local_nnz):
    partition_dict = {}
    
    if isTranspose == 0:
        for edge in partition:
            u, v, data = edge
            if u not in partition_dict:
                partition_dict[u] = []
            partition_dict[u].append((v, data))
    else:
        for edge in partition:
            u, v, data = edge
            if v not in partition_dict:
                partition_dict[v] = []
            partition_dict[v].append((u, data))

    ia = [0]
    ja = []
    jval = []
    #print(len(l2g_map))

    for node in l2g_map:
        u, _ = node
        for edge in partition_dict[u]:
            ja.append(edge[0])
            jval.append(edge[1])
        ia.append(len(ja))

    """
    print("PartID: ", part_id)
    print("Ia: ", len(ia))
    print("Ja: ", len(ja))
    print("nnz: ", local_nnz)
    print("Locals: ", len(locals))
    print("Nonlocals: ", len(nonlocals))
    """
    assert len(ia) == (local_m + 1)
    assert len(ja) == local_nnz
    assert len(jval) == local_nnz

    return ia, ja, jval

def set_local_csr(partition,
                  part_id,
                  locals,
                  nonlocals,
                  l2g_map,                  
                  little_m,
                  local_m,
                  local_nnz):
    partition_dict = {}
    
    for edge in partition:
        u, v, data = edge
        if u not in partition_dict:
            partition_dict[u] = []
        partition_dict[u].append((v, data))
        
    ia = [0]
    ja = []
    jval = []
    #print(len(l2g_map))

    for node in l2g_map:
        u, _ = node
        for edge in partition_dict[u]:
            ja.append(edge[0])
            jval.append(edge[1])
        ia.append(len(ja))

    """
    print("PartID: ", part_id)
    print("Ia: ", len(ia))
    print("Ja: ", len(ja))
    print("nnz: ", local_nnz)
    print("Locals: ", len(locals))
    print("Nonlocals: ", len(nonlocals))
    """
    assert len(ia) == (local_m + 1)
    assert len(ja) == local_nnz
    assert len(jval) == local_nnz

    return ia, ja, jval

def set_l2gMap(partition, 
               part_id,
               vtx_part): 
    locals = set()
    nonlocals = set()
    uniques = set()

    for edge in partition:
        u, v, data = edge
        if vtx_part[u] == vtx_part[v] == part_id:
        #if vtx_part[u] == part_id:
            if u not in uniques:
                uniques.add(u)
                locals.add((u, data))
        else:        
            if vtx_part[u] != part_id:
                if u not in uniques:
                    uniques.add(u)
                    nonlocals.add((u, vtx_part[u], data))
            #elif vtx_part[u] != part_id:
            #    if v not in uniques:
            #        uniques.add(v)
            #        nonlocals.add((v, vtx_part[v], data))
            
    locals = sorted(locals, key=lambda x: x[0])
    #print(locals)


    nonlocals = sorted(nonlocals, key=lambda x: (x[1], x[0]))
    nonlocals = [(x[0], x[2]) for x in nonlocals]
    #print(nonlocals)
    #print("Locals: ", locals)

    return locals, nonlocals 

def set_partitions(adj, 
                   vtx_part,
                   edge_dict,
                   diagonal_data,
                   npart):
    start = time.time()
    partitions = [[] for i in range(npart)]

    for i in range(adj.shape[0]):
        start_idx = adj.indptr[i]
        end_idx = adj.indptr[i + 1]
        for j in range(start_idx, end_idx):
            v = adj.indices[j]
            data = adj.data[j]
            partitions[edge_dict[f"{i} {v}"]].append((i, v, data))
            partitions[edge_dict[f"{v} {i}"]].append((v, i, data))
    for i in range(adj.shape[0]):
        partitions[edge_dict[f"{i} {i}"]].append((i, i, diagonal_data[i]))

    print(f"Partitioning completed! Time: {time.time() - start}")
    
    return partitions

def set_edge_dict(csr, epart, vpart):
    start = time.time()
    edge_dict = {}
    idx = 0
    for i in range(csr.shape[0]):
        start_idx = csr.indptr[i]
        end_idx = csr.indptr[i + 1]
        row_indices = csr.indices[start_idx:end_idx]
        for j in row_indices:
            edge_dict[f"{i} {j}"] = epart[idx]
            edge_dict[f"{j} {i}"] = epart[idx]
            idx += 1

    for i in range(csr.shape[0]):
        edge_dict[f"{i} {i}"] = vpart[i]   

    print(f"Edge dict created! Time: {time.time() - start}")

    return edge_dict

def read_mm(filepath):
    graph_info = mminfo(filepath)
    n_vtx = graph_info[0]
    n_edge = graph_info[2]
    coo = mmread(filepath)
    csr = coo.tocsr()
    diagonal_data = csr.diagonal()
    directed_csr = triu(csr, k=1).tocsr()
    """rows, cols = csr.nonzero() 
    non_self_loops = rows != cols 
    csr_no_self_loop = sp.csr_matrix((csr.data[non_self_loops], 
                                    (rows[non_self_loops], cols[non_self_loops])),
                                    shape=csr.shape)"""
    n_edge = (n_edge - n_vtx)
    print("Graph is read.")
    return directed_csr, diagonal_data, graph_info

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-g', type=str, required=True, help="Path of the matrix(graph) file")
    parser.add_argument('--parts', '-e', type=str, required=True, help="Path of the partitions file (the output bin file is created there)")
    parser.add_argument('--npart', '-p', type=int, required=True, help="Number of partitions")

    args = parser.parse_args()
    return args

def main(args) -> None:
    filename = args.input
    vp_path = args.parts + f"vpart{args.npart}.txt"
    ep_path = args.parts + f"epart{args.npart}.txt"
    vtx_part: list[int] = read_inparts(vp_path)
    print("VP read completed!")
    edge_part: list[int] = read_inparts(ep_path) 
    print("EP read completed!")
    #adj, info = read_symmetric_adj(filename)
    adj, diagonal_data, info = read_mm(filename)
    
    #edge_dict: dict = create_edge_dict(adj, edge_part)
    edge_dict: dict = set_edge_dict(adj, edge_part, vtx_part)
    partitions: list[list[tuple]] = set_partitions(adj, vtx_part, edge_dict, diagonal_data, args.npart)

    partition_list: list[Partition] = []
    for part_id in tqdm(range(args.npart)):
        locals:list[int]
        nonlocals:list[int]
        locals, nonlocals = set_l2gMap(partition=partitions[part_id], 
                                     part_id=part_id, 
                                     vtx_part=vtx_part)
        
        little_m = len(locals)
        local_m = len(locals) + len(nonlocals)
        local_nnz = len(partitions[part_id])
        l2g_map = [*locals, *nonlocals]
        #print("L2gMap: ",l2g_map)
        
        ia, ja, jval = set_local_csr(partition=partitions[part_id],
                        part_id=part_id,
                        locals=locals,
                        nonlocals=nonlocals,
                        l2g_map=l2g_map,
                        little_m=little_m,
                        local_m=local_m,
                        local_nnz=local_nnz)
    
        l2g_map = [x[0] for x in l2g_map]
        part = Partition(local_m=local_m,
                         local_nnz=local_nnz,
                         little_m=little_m,
                         ia=ia,
                         ja=ja,
                         jval=jval,
                         l2g_map=l2g_map)
        
        partition_list.append(part)

    global_m = info[0]
    global_n = info[1]
    global_nnz = info[2]*2 - info[0]
    #n_parts = N_PARTITIONS
    bin_path = args.parts + f"{args.npart}.bin"
    Converter(partition_list=partition_list, 
              global_m=global_m, 
              global_n=global_n, 
              global_nnz=global_nnz, 
              n_parts=args.npart).writeBIN(bin_path)
    
    return


        
class Partition:
    def __init__(self, local_m:int, local_nnz:int, little_m:int, ia, ja, jval, l2g_map):
        self.local_m = local_m
        self.local_nnz = local_nnz
        self.little_m = little_m
        self.ia = ia
        self.ja = ja
        self.jval = jval
        self.l2g_map = l2g_map

if __name__ == '__main__':
    args = get_args()
    main(args)