from concurrent.futures import thread
import sys
from typing import final

import torch
def sys_appned(path: list):
    for sub_path in path:
        if sub_path not in sys.path:
            sys.path.append(sub_path)
    # print("sys.path: \n", sys.path)
# sys_appned(["/mnt/d/CodingArea/Python/TestProejct", "/mnt/d/CodingArea/Python/APPNP/src"])


from uselessCode import ppr_matrix_comparsion, row_node_difference
import threading
import queue
from train import main_APPN
# from main_appnp import main

matrix_queue = queue.Queue()
# print(id(matrix_queue))
barrier = threading.Barrier(2)
lock = threading.Lock()

def get_queue():
    global matrix_queue, barrier, lock
    return barrier, lock

if __name__ == "__main__":
    tgnn_thread = threading.Thread(target=main_APPN, args=(matrix_queue, True))
    appnp_thread = threading.Thread(target=main, args=(True, matrix_queue, True))

    appnp_thread.start()
    tgnn_thread.start()
    appnp_thread.join()
    tgnn_thread.join()

    tgnn_src, tgnn_layer2, tgnn_propagator, tgnn_final = [], [], [], []
    appnp_src, appnp_layer2, appnp_propagator, appnp_final = [], [], [], []
    tgnn = [tgnn_src, tgnn_layer2, tgnn_propagator, tgnn_final]
    appnp = [appnp_src, appnp_layer2, appnp_propagator, appnp_final]

    def name_record(matrix, suffix: str, stat: list[list[torch.Tensor]]):
        src, layer2, propagator, final_ = stat
        if "src" == suffix:
            src.append(matrix)
        elif "layer2" == suffix:
            layer2.append(matrix)
        elif "propagator" == suffix:
            propagator.append(matrix)
        else:
            final_.append(matrix)

    print("matrix_queue size:", matrix_queue.qsize())
    while not matrix_queue.empty():
        status, matrix = matrix_queue.get()
        prefix, name = status.split("_")
        if "tgnn" == prefix:
            name_record(matrix, name, tgnn)
        elif "src" == prefix:
            name_record(matrix, name, appnp)
        else:
            raise ValueError("Invalid prefix")
    
    assert len(tgnn_src) == len(tgnn_final) == len(tgnn_layer2) and \
            len(appnp_layer2) == len(appnp_src) == len(appnp_final), \
            f"threading running in error, order and number not is not equal, \
            {len(tgnn_src)}, {len(appnp_src)}, {len(tgnn_layer2)}, {len(appnp_layer2)}, {len(tgnn_final)}, {len(appnp_final)}"
    
    print(len(tgnn_src), "the type within length is:", type(tgnn_src[0]))

    for idx, (tgnns, appnps) in enumerate(zip(tgnn, appnp)):
        for sub_tgnn, sub_appnp in zip(tgnns[:5], appnps[:5]):
            print(f"#-------------------------Thread checking under layer {idx}-------------------------------")
            print(f"sub_tgnn shape: {sub_tgnn.shape}, sub_appnp shape: {sub_appnp.shape}")
            
            qualify_row, coverage, weight_eval_row, matched_weight_diff, eq_weight\
                = ppr_matrix_comparsion(sub_tgnn, sub_appnp)
    
            # for node difference analysis
            print("\n\nTGNN-PPR and APPNP-PPR comparison for node differnce:")
            print(f"qualified nodes: {qualify_row.shape[0]} \ncoverage: {coverage[coverage>0.1].shape[0]} \
                    \nmax matched sub-list length: {torch.max(coverage[coverage>0.1]).item():03f} \ntotal nodes: {sub_tgnn.shape[0]}")
            
            # for weight difference analysis
            avg_weight = torch.mean(weight_eval_row).item()
            topk5 = torch.topk(weight_eval_row[weight_eval_row>=0.], 5, largest=False)[0]
            # assert len(matched_weight_diff) == coverage[coverage>0.1].shape[0], "Compute node and weight is not equal"

            print("\nTGNN-PPR and APPNP-PPR comparison for weight differnce:")
            print(f"average matrix weight difference: {avg_weight:03f} \nfirst 5 fit weight difference: {[round(val.item(), 5) for val in topk5]} \
                    \nnumber of full weight equaled node: {eq_weight.shape[0]}\
                    \n#--------------------------------------------------------------\n")
            