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

from ShangRu_202307_Test.utils import print_nvidia_smi, \
    set_reproducibility, seed_worker # ShangRu_202307_Test
# -----------------------------------------------------------------------------/

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def main():
    parser = argparse.ArgumentParser(description='Mito-Net')
    # Datasets parameters
    parser.add_argument('--data_path', type=str, default='data/Z_stack', help="root path to data directory")
    parser.add_argument('--workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--target_image', type=str, default='Mito', help='which target to be predicted')
    parser.add_argument('--source_image', type=str, default='TL', help='which source to be trained')

    # Training parameters
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epoch', default=200, type=int, help="number of training iterations")
    parser.add_argument('--val_epoch', default=1, type=int, help="number of validation iterations")
    parser.add_argument('--train_batch', default=2, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)

    parser.add_argument('--model', type=str, default='DenT')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoints', type=str, default='checkpoints')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--seg_dir', type=str, default='seg_results') # ShangRu_202307_Test

    # deep supervision
    parser.add_argument('--deep_supervision', type=bool, default=False)
    # Patches
    parser.add_argument('--patch', type=bool, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    ''' Setup GPU '''
    torch.cuda.set_device(args.gpu)
    # torch.cuda.empty_cache() # ShangRu_202307_Test

    ''' Setup Random Seed '''
    set_reproducibility(args.random_seed) # ShangRu_202307_Test
    g = torch.Generator() # ShangRu_202307_Test
    g.manual_seed(0) # ShangRu_202307_Test
    seed = np.random.randint(100000)

    if not os.path.exists(os.path.join(args.checkpoints, '{}_{}_{}'.format(args.model, args.target_image, seed))):
        os.makedirs(os.path.join(args.checkpoints, '{}_{}_{}'.format(args.model, args.target_image, seed)))
    if not os.path.exists(os.path.join(args.log_dir, 'Train_info_{}_{}_{}'.format(args.model, args.target_image, seed))):
        os.makedirs(os.path.join(args.log_dir, 'Train_info_{}_{}_{}'.format(args.model, args.target_image, seed)))

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
    else:
        raise NotImplementedError

    '''Data Parallel'''
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    ''' Define Loss '''
    criterion = None
    criterion = BCEDiceLoss() #BCEDiceLoss() #nn.CrossEntropyLoss()

    ''' Setup Optimizer '''
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    ''' Setup Tensorboard '''
    writer = SummaryWriter(os.path.join(args.log_dir, 'Train_info_{}_{}_{}'.format(args.model, args.target_image, seed)))
    
    ''' Train Model '''
    print('===> Start training ...\n') # ShangRu_202307_Test
    iters = 0
    best_mIoU = 0
    best_val_loss = 1
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
                save_model(model, os.path.join(args.checkpoints, '{}_{}_{}'.format(args.model, args.target_image, seed), 'model_{}_best_pth.tar'.format(args.model)))
                #best_mIoU = mIoU
                best_val_loss = val_loss
                best_epoch = epoch
        
        ''' Save Model (Define in above)'''
        if epoch % 50 == 0:
            save_model(model, os.path.join(args.checkpoints, '{}_{}_{}'.format(args.model, args.target_image, seed), 'model_{}_{}_pth.tar'.format(args.model, epoch)))
    
    print('The best model occurred in Epoch [{}] with validation loss {}'.format(best_epoch, best_val_loss))

    print(); print_nvidia_smi() # ShangRu_202307_Test
    # writer.add_graph(model, imgs) # ShangRu_202307_Test ( 改成 evaluate 再存 )

if __name__ == '__main__':
	main()
 