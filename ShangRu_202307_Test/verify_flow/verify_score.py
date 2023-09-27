import os
import sys
from pathlib import Path
from colorama import Fore, Back, Style

import numpy as np
import skimage

abs_module_path = Path("/home/twsqzqy988/DenT/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from data import resize_3D
from utils import iou_score
# -----------------------------------------------------------------------------/


def cli_divide_line():
    """
    """
    print(f"\n{Fore.GREEN}{'='*100}{Style.RESET_ALL}")
    # -------------------------------------------------------------------------/
    


def get_file_id(path:Path):
    """
    """
    file_name: str = str(path).split(os.sep)[-1]
    file_id: int = int(file_name.split(".")[0])
    
    return file_id
    # -------------------------------------------------------------------------/



def dump_info(img, desc):
    """
    """
    print("\n{} :".format(desc))
    print("shape: {}".format(img.shape))
    print("dtype: {}".format(img.dtype))
    print("range: ({}, {})".format(np.min(img), np.max(img)))
    # -------------------------------------------------------------------------/



def read_and_preprocess_img(img_path:Path, desc:str):
    """
    """
    print(f"\n{Fore.YELLOW}{desc}_path: '{img_path}'{Style.RESET_ALL}")
    
    img = skimage.io.imread(img_path)
    dump_info(img, desc)
    
    """ Resize """
    c, h, w = img.shape
    if (h == w) and (h == 917):
        img = resize_3D(img, 0.2792) # 917 -> 256 (2792)
        desc += " --> resize"
    
    """ Re-scale to (0.0, 1.0) """
    img = img/255
    desc += " --> range: (0.0, 1.0)"
    
    """ Dump after preprocess """
    dump_info(img, f">>> {desc}")
    
    return img
    # -------------------------------------------------------------------------/



if __name__ == '__main__':
    
    # pred_dir
    dent_pred_dir: Path = Path(r"/home/twsqzqy988/DenT/ShangRu_202307_Test/verify_flow/result (NAS, Chan-Min Hsu)/IoU/New Model")
    
    # gt_dir
    dent_gt_dir: Path = Path(r"/home/twsqzqy988/DenT/data/{DataSet}_DenT/test")
    dent_gt_dir = dent_gt_dir.joinpath("target") # target / target_dna

    """ Get paths """
    pred_paths = sorted(dent_pred_dir.glob(f"*.tif"), key=get_file_id) # (32, 256, 256)
    gt_paths = sorted(dent_gt_dir.glob(f"*.tif"), key=get_file_id) # (32, 917, 917)
    assert len(pred_paths) == len(gt_paths), "len(pred_paths) != len(gt_paths)"
    
    """ Prepare images """
    pred_img_list: list = []
    gt_img_list: list = []
    for i, (pred_path, gt_path) in enumerate(zip(pred_paths, gt_paths)):
        
        cli_divide_line(); print(f"{Fore.MAGENTA}[{i+1}]{Style.RESET_ALL}")
        assert str(pred_path).split(os.sep)[-1] == \
            str(gt_path).split(os.sep)[-1], "file_name not match"

        pred_img = read_and_preprocess_img(pred_path, "pred")
        pred_img_list.append(pred_img)
        
        gt_img = read_and_preprocess_img(gt_path, "gt")
        gt_img_list.append(gt_img)
    
    """ Calculate score """
    preds = np.concatenate(pred_img_list)
    gts = np.concatenate(gt_img_list)
    cli_divide_line()
    meanIoU = iou_score(preds, gts)
    
    sys.exit()