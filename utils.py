import numpy as np
import scipy.misc
import argparse
import os
import glob
import torch
import torch.nn as nn # ShangRu_202307_Test
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + 0.5 * dice

def read_masks(seg_dir, seg_h=352, seg_w=448):
    '''
    Read masks from directory
    '''
    seg_paths = glob.glob(os.path.join(seg_dir,'*.png'))
    seg_paths.sort()
    segs = np.zeros((len(seg_paths), seg_h, seg_w))
    for idx, seg_path in enumerate(seg_paths):
        seg = scipy.misc.imread(seg_path)
        segs[idx, :, :] = seg
    
    return segs

def iou_score(output, target):
    smooth = 1e-5

    output_ = output > 0.5 # True or False
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    print('mean_iou: %f\n' % iou)

    return iou

def mean_iou_score(pred, labels, num_classes=1):
    '''
    Compute mean IoU score over 9 classes
    '''
    mean_iou = 0
    for i in range(num_classes):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / num_classes
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
    parser.add_argument('-p', '--pred', help='prediction masks directory', type=str)
    args = parser.parse_args()

    pred = read_masks(args.pred)
    labels = read_masks(args.labels)
        
    print('# preds: {}; pred.shape: {}'.format(len(pred), pred.shape))
    print('# labels: {}; labels.shape: {}'.format(len(labels), labels.shape))
    mean_iou_score(pred, labels)
