from medpy.metric.binary import assd,dc
import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pred_pth = './result'
gt_pth = './data/testB/labels'
filenames = os.listdir(pred_pth)

def one_hot(seg):
    vals = np.array([0, 1, 2, 3, 4])
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for i, c in enumerate(vals):
        res[i][seg == c] = 1
    return res
def main():
    labels_tr_1 = np.zeros((256, 256, 256))
    output_tr_1 = np.zeros((256, 256, 256))
    labels_tr_2 = np.zeros((256, 256, 256))
    output_tr_2 = np.zeros((256, 256, 256))
    labels_tr_3 = np.zeros((256, 256, 256))
    output_tr_3 = np.zeros((256, 256, 256))
    labels_tr_4 = np.zeros((256, 256, 256))
    output_tr_4 = np.zeros((256, 256, 256))
    for name in filenames:
        if 'image' in name:
            if '1003' in name:
                pred = Image.open(os.path.join(pred_pth, name))
                gt_name = name.replace('image', 'gth')
                gt = Image.open(os.path.join(gt_pth, gt_name))
                pred = np.asarray(pred, np.float32)
                gt = np.asarray(gt, np.float32)
                index = int(name.replace('.png', '')[18:])
                output_tr_1[index] = pred
                labels_tr_1[index] = gt
    for name in filenames:
        if 'image' in name:
            if '1008' in name:
                pred = Image.open(os.path.join(pred_pth, name))
                gt_name = name.replace('image', 'gth')
                gt = Image.open(os.path.join(gt_pth, gt_name))
                pred = np.asarray(pred, np.float32)
                gt = np.asarray(gt, np.float32)
                index = int(name.replace('.png', '')[18:])
                output_tr_2[index] = pred
                labels_tr_2[index] = gt
    for name in filenames:
        if 'image' in name:
            if '1014' in name:
                pred = Image.open(os.path.join(pred_pth, name))
                gt_name = name.replace('image', 'gth')
                gt = Image.open(os.path.join(gt_pth, gt_name))
                pred = np.asarray(pred, np.float32)
                gt = np.asarray(gt, np.float32)
                index = int(name.replace('.png', '')[18:])
                output_tr_3[index] = pred
                labels_tr_3[index] = gt
    for name in filenames:
        if 'image' in name:
            if '1019' in name:
                pred = Image.open(os.path.join(pred_pth, name))
                gt_name = name.replace('image', 'gth')
                gt = Image.open(os.path.join(gt_pth, gt_name))
                pred = np.asarray(pred, np.float32)
                gt = np.asarray(gt, np.float32)
                index = int(name.replace('.png', '')[18:])
                output_tr_4[index] = pred
                labels_tr_4[index] = gt

    output_tr = np.concatenate((output_tr_1, output_tr_2, output_tr_3, output_tr_4), axis=0)
    labels_tr = np.concatenate((labels_tr_1, labels_tr_2, labels_tr_3, labels_tr_4), axis=0)
    output_tr = output_tr / 51
    labels_tr = labels_tr / 51
    output_tr = one_hot(output_tr)
    labels_tr = one_hot(labels_tr)
    output_tr = np.transpose(output_tr, (1, 0, 2, 3))
    labels_tr = np.transpose(labels_tr, (1, 0, 2, 3))
    dice_score_total = []
    assd_score_total = []
    for c in range(1, 5):
        dice_score_total.append(dc(output_tr[:, c], labels_tr[:, c]))
        assd_score_total.append(assd(output_tr[:, c], labels_tr[:, c]))
    print('DICE SCORE : AA {}'.format(dice_score_total[3]))
    print('DICE SCORE : LAC {}'.format(dice_score_total[1]))
    print('DICE SCORE : LVC {}'.format(dice_score_total[2]))
    print('DICE SCORE : MYO {}'.format(dice_score_total[0]))
    print('DICE SCORE : Average {}'.format(np.mean(dice_score_total)))
    print('--------------------------------------------')
    print('ASSD SCORE : AA {}'.format(assd_score_total[3]))
    print('ASSD SCORE : LAC {}'.format(assd_score_total[1]))
    print('ASSD SCORE : LVC {}'.format(assd_score_total[2]))
    print('ASSD SCORE : MYO {}'.format(assd_score_total[0]))
    print('ASSD SCORE : Average {}'.format(np.mean(assd_score_total)))
    return dice_score_total


if __name__ == '__main__':
    main()