import os
import torch

import argparse
import DenT
import data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from test import evaluate
from utils import BCEDiceLoss

import warnings
warnings.filterwarnings("ignore", '.*output shape of zoom.*')

# ShangRu_202307_Test
import sys
from pathlib import Path
from datetime import datetime

abs_module_path = Path("/work/twsqzqy988/DenT-PaperRevision").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.insert(0, str(abs_module_path)) # add path to scan customized module

from misc_utils import print_nvidia_smi, \
    set_reproducibility, seed_worker, get_args, set_args_dirs, dump_config
# -----------------------------------------------------------------------------/

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def main():
    
    args = get_args("train")  # ShangRu_202307_Test

    ''' Setup GPU '''
    # torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache() # ShangRu_202307_Test

    ''' Setup Random Seed '''
    set_reproducibility(args.random_seed) # ShangRu_202307_Test
    g = torch.Generator() # ShangRu_202307_Test
    g.manual_seed(0) # ShangRu_202307_Test
    seed = np.random.randint(100000)

    ''' Set dirs '''
    set_args_dirs(args, seed, "train") # ShangRu_202307_Test
    time_stamp: str = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    print(f"datetime: {time_stamp}\n")
    dump_config(Path(args.log_dir).joinpath(f"{time_stamp}_args.toml"), args.__dict__) # ShangRu_202307_Test

    ''' Load Dataset and Prepare Dataloader '''
    print('===> Preparing dataloader ... ')
    train_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               worker_init_fn=seed_worker, # ShangRu_202307_Test
                                               generator=g, # ShangRu_202307_Test
                                               pin_memory=True, # ShangRu_202307_Test
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='val'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             pin_memory=True, # ShangRu_202307_Test
                                             shuffle=False)

    ''' Load Model '''
    print('===> Preparing model ...')
    model = None
    if args.model == 'DenT':
        model = DenT.DenseTransformer(args)
    elif args.model == 'CusDenT':
        model = DenT.CustomizableDenT(add_pos_emb=args.add_pos_emb,
                                      use_multiheads=[bool(m) for m in args.use_multiheads])
    else:
        raise NotImplementedError

    '''Data Parallel'''
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    ''' Define Loss '''
    criterion = None
    criterion = BCEDiceLoss() #nn.BCEWithLogitsLoss() #BCEDiceLoss() #nn.CrossEntropyLoss()

    ''' Setup Optimizer '''
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    ''' Setup Tensorboard '''
    writer = SummaryWriter(args.log_dir) # ShangRu_202307_Test
    
    ''' Train Model '''
    print('===> Start training ...\n') # ShangRu_202307_Test
    iters = 0
    best_mIoU = 0
    best_val_loss = np.inf # ShangRu_202307_Test
    for epoch in range(1, args.epoch + 1):
        model.train()
        for idx, (img_ids, imgs, segs) in enumerate(train_loader): # ShangRu_202307_Test
            train_info = 'Epoch: [{0}][{1}/{2}], [{3}]'.format(epoch, idx+1, len(train_loader), img_ids) # ShangRu_202307_Test
            iters += 1

            if args.patch == True:
                imgs = imgs.contiguous().view(-1, 1, 32, 256, 256)
                segs = segs.contiguous().view(-1, 1, 32, 256, 256)
                        
            imgs, segs = imgs.cuda(), segs.cuda()
            
            if args.deep_supervision:
                outputs = model(imgs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, segs)
                loss /= len(outputs)

            else:
                output = model(imgs)
                loss = None
                loss = criterion(output, segs)
            
            optimizer.zero_grad()   # set grad of all parameters to zero
            loss.backward()         # compute gradient for each parameters
            optimizer.step()        # update parameters
            
            ''' Write Out Information to Tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:4f}'.format(loss.data.cpu().numpy())
            print(train_info)
        
        if epoch % args.val_epoch == 0:
            # Evaluate the Model
            with torch.no_grad():
                mIoU, val_loss = evaluate(args, model, val_loader)
            writer.add_scalar('val_mIoU', mIoU, epoch)
            print('Epoch [{}]: mean IoU: {}'.format(epoch, mIoU))
            writer.add_scalar('val_loss', val_loss.data.cpu().numpy(), epoch)
            print('Epoch [{}]: validation loss: {}'.format(epoch, val_loss.data.cpu().numpy()))
            
            print(); print("="*80, "\n") # ShangRu_202307_Test

            # Save Best Model 
            if val_loss < best_val_loss - 1e-4:
                save_model(model, Path(args.checkpoints).joinpath(f"model_{args.model}_best_pth.tar")) # ShangRu_202307_Test
                #best_mIoU = mIoU
                best_val_loss = val_loss
                best_epoch = epoch
        
        ''' Save Model (Define in above)'''
        if epoch % 50 == 0:
            save_model(model, Path(args.checkpoints).joinpath(f"model_{args.model}_{epoch}_pth.tar")) # ShangRu_202307_Test
    
    print('The best model occurred in Epoch [{}] with validation loss {}'.format(best_epoch, best_val_loss))
    print()
    print_nvidia_smi() # ShangRu_202307_Test
    # writer.add_graph(model, imgs) # ShangRu_202307_Test ( 改成 evaluate 再存 )
    # -------------------------------------------------------------------------/



if __name__ == '__main__':
	main()