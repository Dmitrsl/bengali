import torch
from catalyst.utils import set_global_seed, prepare_cudnn, get_device
import gc
import numpy as np
import multiprocessing

def settings():

    GPU_TRAIN = torch.cuda.is_available()
    SEED = 2020
    FP16 = True
    NUM_CORES = multiprocessing.cpu_count()
    BS = 8
    if GPU_TRAIN:
        CUDA_NAME = torch.cuda.get_device_name()
        FP16 = True
        if CUDA_NAME in ['Tesla K80','Tesla P4']:
            BS = 32
        else:
            BS = 64
        if FP16:
            BS = int(BS*2)
        print(f'GPU: {CUDA_NAME}')

    np.random.seed(SEED)
    #random.seed(SEED)
    set_global_seed(SEED)
    prepare_cudnn()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Number of cores CPU: {NUM_CORES}')
    print(f'Batch size: {BS}')
    print(torch.cuda.get_device_properties(device).total_memory / 2 ** 20)
    
    
    return NUM_CORES -2, BS

if __name__ == '__main__':
    settings()