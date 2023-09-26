import subprocess # ShangRu_202307_Test

import random
import numpy as np
import torch
# -----------------------------------------------------------------------------/


def print_nvidia_smi():
    """ show `nvidia-smi`
    """ 
    # 要執行的命令
    command = "nvidia-smi"
    # 使用Popen執行命令，將stdout捕獲到PIPE中
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 從stdout讀取輸出
    output, error = process.communicate()
    # 將bytes轉換為字符串
    output_str = output.decode('utf-8')
    # 輸出結果
    print(f"{output_str}")
    # -------------------------------------------------------------------------/



def set_reproducibility(seed):
    """ Pytorch reproducibility
        - ref: https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/?amp=1
        - ref: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # current GPU
    torch.cuda.manual_seed_all(seed) # all GPUs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # -------------------------------------------------------------------------/



def seed_worker(worker_id):
    """ DataLoader reproducibility
        ref: 'https://pytorch.org/docs/stable/notes/randomness.html'
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # -------------------------------------------------------------------------/