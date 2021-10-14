# detector entity
import os
import mmcv
from sys import path
path.append(r'../')
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector, show_result
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from detector.config import detector as detect_cfg
from detector.detector_util import mask_detect_result
from data.image.image_entity import Image
from data.image.config import image as image_config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Detector(object):
    def __init__(self, detector_cfg=None, is_pretrained=False):
        self.detector_name = detector_cfg.detector['name']
        self.detector = None
        self.is_pretrained = is_pretrained
        
        assert detector_cfg is not None
        self.detector_cfg = detector_cfg
        self.cfg = detector_cfg.detector['cfg_list'][detector_cfg.detector['name_list'].index(self.detector_name)]
        print('[detector] load {}, url {} \n, cfg_path {}'.format(self.detector_name, self.cfg['url'], self.cfg['cfg_path']))
        

    def detect_all(self, image: Image):
        for img in image.sample():
            result = self.detect(img)
            result = mask_detect_result(self.detector.CLASSES, result, image.crop_size[0], image.crop_size[1])
            image.get_result(result)
        if image_config['multi_scale']:
            for img in image.ms_sample():
                result = self.detect(img)
                result = mask_detect_result(self.detector.CLASSES, result, image.crop_size[0], image.crop_size[1])
                image.get_result_ms(result)
        image.nms_all_result()
        # image.draw()
        return image

    def detect(self, img):
        if img is None:
            print("[detector error] image not found")
        with torch.no_grad():
        #x = Variable(img.unsqueeze(0))
            detect_result = inference_detector(self.detector, img)

        #if detect_cfg['cuda']:
        #    x = x.cuda()

        #detect_result = self.detector(x).data

        return detect_result

    def load_detector(self):
        if detect_cfg['cuda']:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cfg = mmcv.Config.fromfile(self.cfg['cfg_path'])
        if not self.is_pretrained:
            cfg.model.pretrained = None
        model = init_detector(cfg, self.cfg['url'], device=torch.device('cuda'))

        # model = build_detector(cfg.model, test_cfg = cfg.test_cfg)
        # load_checkpoint(model, self.cfg['url'])

        #dataset = build_dataset(cfg.data.test)
        #model.CLASSES = dataset.CLASSES

        model.eval()
        #model = build_detector(cfg.model, test_cfg = cfg.test_cfg)
        #load_checkpoint(model, self.cfg['url'])

        self.detector = model
        #self.detector.eval()
        self.detector_cfg = cfg
        """
        if self.detector_name == 'ssd300':
            self.detector = ssd.build_ssd('test', 300, 21)
        if self.detector_name == 'ssd300_vgg':
            self.detector = ssd_vgg.build_ssd('test', 300, 21)
        self.detector.load_state_dict(torch.load(self.trained_path))
        self.detector.eval()
        print("[detector log] successfully load detector: " + detector['name'])
        if detector['cuda']:
            self.detector = self.detector.cuda()
            cudnn.benchmark = True
        """

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    image = Image(dataset_name='1-HIT_Canteen', frame_name='IMG_1_15')

    image.cv2_resize()
    image.cv2_toNumpy()
    #image.normalize()
    #image.hwc2chw()
    detector = Detector(is_pretrained=True)
    detector.load_detector()
    detector.detect_all(image)

