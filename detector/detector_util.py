import torch
import numpy as np
from detector.config import detector as det_cfg
#from .ssd.data.voc0712 import VOC_CLASSES

def mask_detect_result(classes, detect_result, w, h, ths=det_cfg['threshold']):
    PERSON_INDEX = classes.index(det_cfg['class_list'])
    #

    dets = np.asarray([x for x in detect_result[PERSON_INDEX] if x[4] > ths])
    if len(dets) > 0:
        boxes = dets[:, :4]
        scores = dets[:,4]
        cls_dets = np.hstack((boxes,
                              scores[:, np.newaxis])).astype(np.float32,
                                                             copy=False)
    else:
        cls_dets = None
    """
    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
    dets = torch.masked_select(dets, mask).view(-1, 5)
    if dets.size(0) == 0:
        return None
    boxes = dets[:, :4]
    # x1 y1 x2 y2 score
    boxes[:,0] *= w
    boxes[:,2] *= w
    boxes[:,1] *= h
    boxes[:,3] *= h
    scores = dets[:,0].cpu().numpy()
    cls_dets = np.hstack((boxes.cpu().numpy(),
                          scores[:, np.newaxis])).astype(np.float32,
                                                         copy=False)
    """
    return cls_dets
