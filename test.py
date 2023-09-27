import os 
import torch
import tifffile

import argparse
import DenT
import data
from utils import *
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path # ShangRu_202307_Test
from ShangRu_202307_Test.utils import print_nvidia_smi, \
    set_reproducibility, seed_worker # ShangRu_202307_Test
# -----------------------------------------------------------------------------/

def reconstruct(pred, seg):
    pred_patch, seg_patch, new_preds, new_segs = list(),list(),list(),list()
    for i in range(len(pred)):
        pred_patch.append(pred[i])
        seg_patch.append(seg[i])    
    for i in range(0,len(pred),2):
        new_preds.append(np.concatenate(pred_patch[i:i+2], axis=2))
        new_segs.append(np.concatenate(seg_patch[i:i+2], axis=2))
    recon_preds = np.expand_dims(np.concatenate(new_preds, axis=1), 0)
    recon_segs = np.expand_dims(np.concatenate(new_segs, axis=1), 0)
    
    return recon_preds, recon_segs

def evaluate(args, model, data_loader, save_img=False):
    ''' set model to evaluate mode '''
    model.eval()
    preds_dict = dict()
    gts_dict = dict()

    criterion = None
    criterion = BCEDiceLoss()
    loss = 0
    count = 0

    with torch.no_grad():
        for idx, (img_names, imgs, segs) in enumerate(data_loader):
            if args.patch == True:
                imgs = imgs.contiguous().view(-1, 1, 32, 256, 256)
                segs = segs.contiguous().view(-1, 1, 32, 256, 256)

            imgs = imgs.cuda()
            segs = segs.cuda()
            preds = model(imgs) #(batch_size, C, H, W)
            
            #evaluate validation loss
            loss += criterion(preds, segs)
            count += 1

            preds = torch.sigmoid(preds).cpu().numpy().squeeze(1)
            segs = segs.cpu().numpy().squeeze(1)
            
            # reconstruct images from patches
            if save_img:
                if args.patch == True:
                    preds, segs = reconstruct(preds, segs)
                for img_name, pred, seg in zip(img_names, preds, segs): 
                    preds_dict[img_name] = pred
                    gts_dict[img_name] = seg  
            else:
                for i in range(len(preds)):
                    preds_dict[str(i)] = preds[i]
                    gts_dict[str(i)] = segs[i]

    gts = np.concatenate(list(gts_dict.values()))
    preds = np.concatenate(list(preds_dict.values()))

    meanIoU = iou_score(preds, gts)
    meanloss = loss / count

    if args.seg_dir != '' and save_img:
        if not os.path.exists(args.seg_dir):
            os.makedirs(args.seg_dir)
        for img_name, pred in preds_dict.items():
            tifffile.imwrite(os.path.join(args.seg_dir, img_name), (pred*255).astype('uint8'))

    return meanIoU, meanloss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mito-Net')
    # Datasets Parameters
    parser.add_argument('--data_path', type=str, default='data/Z_stack_2D', help="root path to data directory")
    parser.add_argument('--workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--target_image', type=str, default='Mito', help='which target to be predicted')
    parser.add_argument('--source_image', type=str, default='TL', help='which source to be trained')

    # Testing parameters
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--test_batch', default=1, type=int)
    parser.add_argument('--model', type=str, default='DenT')
    parser.add_argument('--checkpoints', type=str, default='checkpoints')
    parser.add_argument('--random_seed', type=int, default=123) # ShangRu_202307_Test
    parser.add_argument('--seg_dir', type=str, default='seg_results')
    parser.add_argument('--patch', type=bool, default=False)

    args = parser.parse_args()

    # seed = 15725 #random_seed: 123 >> 15725 # ShangRu_202307_Test
    set_reproducibility(args.random_seed) # ShangRu_202307_Test
    seed = np.random.randint(100000) # ShangRu_202307_Test
    assert seed == 15725 # ShangRu_202307_Test

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    # torch.cuda.empty_cache() # ShangRu_202307_Test
    
    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='test'),
                                         batch_size=args.test_batch,
                                         num_workers=args.workers,
                                         pin_memory=True, # ShangRu_202307_Test
                                         shuffle=False)
    
    ''' prepare best model for visualization and evaluation '''
    model = None
    if args.model == 'DenT':
        model = DenT.DenseTransformer(args)
    else:
        raise NotImplementedError

    '''Data Parallel'''
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    
    best_model_path = Path(os.path.join(args.checkpoints, # ShangRu_202307_Test
                                        # '{}_{}_{}'.format(args.model, args.target_image, seed), # ShangRu_202307_Test
                                        'model_{}_best_pth.tar'.format(args.model)))
    best_checkpoint = torch.load(best_model_path); print(f"load model : {best_model_path.resolve()}") # ShangRu_202307_Test
    model.load_state_dict(best_checkpoint)
    iou, loss = evaluate(args, model, test_loader, save_img=True)
    print('Testing iou: {}'.format(iou))
    print('Testing loss: {}'.format(loss))
    
    print(); print_nvidia_smi() # ShangRu_202307_Test
