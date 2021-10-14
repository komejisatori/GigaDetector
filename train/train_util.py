import numpy as np
import os
from .config import eval

def draw_fppi_mr(results, gts, filenames):
    save_path = eval['save_path']
    img_name_path = os.path.join(save_path, 'name.txt')
    with open(img_name_path, 'w') as f:
        for i in filenames:
            f.writelines(i+'.jpg\n')
    gt_name_path = os.path.join(save_path, 'gt.txt')
    with open(gt_name_path, 'w') as f:
        for single_image_gt in gts:
            for i in single_image_gt:
                f.write(str(i[0]) + ',')
                f.write(str(i[1]) + ',')
                f.write(str(i[2] - i[0]) + ',')
                f.write(str(i[3] - i[1]) + ';')
            f.write('\n')

    anno_path = os.path.join(save_path, 'anno.txt')
    score_path = os.path.join(save_path, 'score.txt')
    with open(score_path, 'w') as f_score:
        with open(anno_path, 'w') as f_anno:
            for single_image_result in results:
                for i in single_image_result:
                    f_anno.write(str(i[0]) + ',')
                    f_anno.write(str(i[1]) + ',')
                    f_anno.write(str(i[2] - i[0]) + ',')
                    f_anno.write(str(i[3] - i[1]) + ';')
                    f_score.write(str(i[4]) + ';')

                f_anno.write('\n')
                f_score.write('\n')
    print('[train util] now run matlab script for png')


def cal_ap(predict, anno, imgnum, ovthresh=eval['ovthresh']):
    
    predict = np.asarray(predict).astype(float)

    
    anno_detected = []
    anno_list = []
    npos = 0
    for a in anno:
        anno_detected.append(np.zeros(len(a)))
        anno_list.append(np.asarray(a).astype(float))
        npos += len(a)

    #anno = np.asarray(anno).astype(float)

    sorted_ind = np.argsort(-predict[:,5])
    # sorted_predict = np.sort(-predict[:,4])
    predict = predict[sorted_ind, :]
    nd = predict.shape[0]
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        index = int(predict[d, 0]) - 1
        bb = predict[d, 1:]
        anno_tmp = anno_list[index]
        ixmin = np.maximum(anno_tmp[:, 0], bb[0])
        iymin = np.maximum(anno_tmp[:, 1], bb[1])
        ixmax = np.minimum(anno_tmp[:, 2], bb[2])
        iymax = np.minimum(anno_tmp[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
               (anno_tmp[:, 2] - anno_tmp[:, 0]) *
               (anno_tmp[:, 3] - anno_tmp[:, 1]) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not anno_detected[index][jmax]:
                tp[d] = 1
                anno_detected[index][jmax] = 1
            else:
                fp[d] = 1
        else:
            fp[d] = 1
            # no difficult tag

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, eval['use_07_metric'])

    return anno_detected

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def cal_iou():
    pass
