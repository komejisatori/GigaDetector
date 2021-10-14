import os
import mmcv
import cv2
import numpy as np
import math
from script.config import *
from data.image.config import image as img_cfg
from data.image.image_util import nms
from detector.config import ssd
import torch

class Image(object):
    def __init__(self, frame_index=0, resize_shape=img_cfg['resize_shape'], dataset_name = None, frame_name=None, ms_resize_shape=img_cfg['ms_resize_scale']):
        """
        read frame from file
        :param frame_index: frame num, start from 0
        :param resize_shape:
        """
        if frame_name is None:
            print("[image log] reading frame %d from video data" % frame_index)
            self.frame_index = frame_index
            self.image_path = os.path.join(img_cfg['frame_path'], img_cfg['dataset_name'], '%d.jpg' % frame_index)
        else:
            self.name = frame_name.split('.')[0]
            if dataset_name is None:
                self.image_path = os.path.join(img_cfg['frame_path'], frame_name + '.jpg')
            print("[image log] reading image {} from {}" .format(frame_name, dataset_name))
            self.image_path = os.path.join(img_cfg['frame_path'], dataset_name, frame_name + '.jpg')
        try:
            self.image = mmcv.imread(self.image_path)
        except:
            self.image = None
        self.image_ms = None
        if self.image is None:
            print("[image error] image not loaded")
        else:
            print("[image log] successfully loaded image")
            self.origin_shape = self.image.shape
            self.resize_shape = resize_shape
            self.ms_resize_shape = ms_resize_shape
            self.crop_size = ssd['shape']
            self.max_row = int(self.resize_shape[0] // ssd['shape'][0] // img_cfg['slide_stride'])
            self.max_col = int(self.resize_shape[1] // ssd['shape'][1] // img_cfg['slide_stride'])
            self.detect_result = [[[] for _ in range(self.max_row)]
                                  for _ in range(self.max_col)]
            self.result = []
            self.all_result = []
            self.cur_row = 0
            self.cur_col = 0

    def cv2_resize(self, for_ms=True):
        self.image = mmcv.imresize(self.image, self.resize_shape, interpolation='area')
        print("[image log] {}, {} resized to {}, {}".format(self.origin_shape[0], self.origin_shape[1],
                                                            self.resize_shape[0], self.resize_shape[1]))
        if img_cfg['multi_scale'] and for_ms:
            print("[image log] add multi_scale, resize scale: {}".format(img_cfg['ms_resize_scale']))
            self.image_ms = self.image[self.resize_shape[1] // 2:, :] #y , x

            self.image_ms = mmcv.imresize(self.image_ms, self.ms_resize_shape, interpolation='area')
            #cv2.imwrite('./1.jpg', self.image_ms)



    def cv2_toNumpy(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = np.array(self.image).astype(np.float32)
        if img_cfg['multi_scale']:
            self.image_ms = cv2.cvtColor(self.image_ms, cv2.COLOR_BGR2RGB)
            self.image_ms = np.array(self.image_ms).astype(np.float32)

    def hwc2chw(self):
        # from hwc to chw
        self.image = self.image.transpose((2, 0, 1))
        if img_cfg['multi_scale']:
            self.image_ms = self.image_ms.transpose((2, 0, 1))

    def normalize(self):
        self.image -= ssd['mean']
        self.image = self.image.astype(np.float32)
        if img_cfg['multi_scale']:
            self.image_ms -= ssd['mean']
            self.image_ms = self.image_ms.astype(np.float32)

    def nms_all_result(self):
        if img_cfg['nms'] :
            self.result = np.array(self.result)
            self.result = torch.from_numpy(self.result)
            keep, count = nms(self.result[:, :4], self.result[:, 4])
            print('[image log] {} boxes to {} boxes'.format(self.result.shape[0], count))
            new_result = []
            keep = keep.numpy().tolist()
            for i in keep[0:count]:
                new_result.append(self.result[i])
            self.result = new_result

    def get_result_ms(self, result):
        if result is None:
            return
        #print('here')
        print('ms {} {} get {} results '.format(self.cur_col, self.cur_row, result.shape[0]))
        col_num = self.resize_shape[1] // (ssd['shape'][1] * img_cfg['slide_stride']) // 2
        base_offset_y = self.resize_shape[1] / 2
        offset_x = self.cur_row * img_cfg['slide_stride'] * self.crop_size[0]
        offset_y = self.cur_col * img_cfg['slide_stride'] * self.crop_size[1]

        scale_x = self.resize_shape[0] / self.ms_resize_shape[0]
        scale_y = (self.resize_shape[1] // 2) / self.ms_resize_shape[1]
        for j in range(result.shape[0]):

            h_div_w = (result[j,3] - result[j,1]) / (result[j,2] - result[j,0])
            if h_div_w < img_cfg['h/w_threshold']:
                continue
            min_offset_left = result[j, 0]
            min_offset_right = self.crop_size[1] - result[j, 2]
            min_offset_up = result[j, 1]
            min_offset_down = self.crop_size[0] - result[j, 3]
            if min_offset_left < img_cfg['min_offset_threshold'] or \
                    min_offset_right < img_cfg['min_offset_threshold'] or \
                    min_offset_down < img_cfg['min_offset_threshold'] or \
                    min_offset_up < img_cfg['min_offset_threshold']:
                continue
            result[j, 0] += offset_x
            result[j, 1] += offset_y
            result[j, 2] += offset_x
            result[j, 3] += offset_y

            result[j, 0] *= scale_x
            result[j, 1] *= scale_y
            result[j, 2] *= scale_x
            result[j ,3] *= scale_y

            result[j, 1] += base_offset_y
            result[j, 3] += base_offset_y


            self.result.append(result[j])



    def get_result(self, result):
        if result is None:
            return
        #print('here')

        offset_x = self.cur_row * img_cfg['slide_stride'] * self.crop_size[0]
        offset_y = self.cur_col * img_cfg['slide_stride'] * self.crop_size[1]
        for j in range(result.shape[0]):

            h_div_w = (result[j,3] - result[j,1]) / (result[j,2] - result[j,0])
            if h_div_w < img_cfg['h/w_threshold']:
                continue
            
            min_offset_left = result[j,0]
            min_offset_right = self.crop_size[1] - result[j,2]
            min_offset_up = result[j,1]
            min_offset_down = self.crop_size[0] - result[j,3]
            if min_offset_left < img_cfg['min_offset_threshold'] or\
                min_offset_right < img_cfg['min_offset_threshold'] or\
                min_offset_down < img_cfg['min_offset_threshold'] or\
                min_offset_up < img_cfg['min_offset_threshold']:
                continue


            result[j, 0] += offset_x
            result[j, 1] += offset_y
            result[j, 2] += offset_x
            result[j, 3] += offset_y


            self.result.append(result[j])

    def __select_result(self, result):
        if result is None:
            return


    def result_to_np(self):
        self.result = np.array(self.result)

    def draw(self, path=None):
        self.result = np.array(self.result)
        color_board = [[255, 0, 0], [0, 255, 255]]
        color_grid = [0,255,0]
        origin_image = cv2.imread(self.image_path)
        origin_image = cv2.resize(origin_image, self.resize_shape)
        for i in range(self.max_col):
            for j in range(self.max_row):
                cv2.rectangle(origin_image,
                              (int(j*img_cfg['slide_stride']*self.crop_size[0]),
                               int(i*img_cfg['slide_stride']*self.crop_size[1])),
                              (int(j * img_cfg['slide_stride'] * self.crop_size[0]) + self.crop_size[0],
                               int(i * img_cfg['slide_stride'] * self.crop_size[1]) + self.crop_size[1]),
                              color_grid, 1
                              )
                '''
                for p in range(len(self.detect_result[i][j])):
                    cv2.rectangle(origin_image,
                                  (int(self.detect_result[i][j][p][0]), int(self.detect_result[i][j][p][1])),
                                  (int(self.detect_result[i][j][p][2]), int(self.detect_result[i][j][p][3])),
                                  color, 2
                                  )
                '''
        color = color_board[0]
        if img_cfg['nms'] and path is None:
            for i in keep.numpy():
                cv2.rectangle(origin_image,
                              (int(self.result[i, 0]), int(self.result[i, 1])),
                              (int(self.result[i, 2]), int(self.result[i, 3])),
                              color, 2
                              )
        else:
            for i in range(self.result.shape[0]):
                cv2.rectangle(origin_image,
                              (int(self.result[i, 0]), int(self.result[i, 1])),
                              (int(self.result[i, 2]), int(self.result[i, 3])),
                              color, 2
                              )
        if path is None:
            cv2.imwrite('result.jpg', origin_image)
        else:
            cv2.imwrite(path, origin_image)



    def sample(self, row=0, col=0):
        """
        return sampled patch
        :param row:
        :param col:
        :return:
        """
        # chw
        self.cur_row = row
        self.cur_col = col
        col_num = self.resize_shape[1] // ssd['shape'][1] - 1 # + 1 for padding-bottom
        if img_cfg['multi_scale']:
            col_num = col_num / 2
        while col <= col_num:
            print("[image log] clip col:{} row:{}".format(self.cur_col, self.cur_row))
            _h1 = int(col*ssd['shape'][1])
            _w1 = int(row*ssd['shape'][0])
            #yield self.image[:, _h1:_h1+ssd['shape'][1], _w1:_w1+ssd['shape'][0]]
            yield self.image[_h1:_h1+ssd['shape'][1], _w1:_w1+ssd['shape'][0],:]
            row += img_cfg['slide_stride']
            self.cur_row += 1
            if _w1+ssd['shape'][0] >= self.resize_shape[0]:
                row = 0
                self.cur_row = 0
                col += img_cfg['slide_stride']
                self.cur_col += 1


    def ms_sample(self, row=0, col=0):
        # self.cur_col -=1
        row = 0
        col = 0
        self.cur_col = 0
        while col <= self.ms_resize_shape[1] // ssd['shape'][1] - 1:
            print("[image log] ms clip col:{} row:{}".format(self.cur_col, self.cur_row))
            _h1 = int(col * ssd['shape'][1])
            _w1 = int(row * ssd['shape'][0])
            #yield self.image_ms[:, _h1:_h1 + ssd['shape'][1], _w1:_w1 + ssd['shape'][0]]
            yield self.image_ms[_h1:_h1 + ssd['shape'][1], _w1:_w1 + ssd['shape'][0], :]
            row += img_cfg['slide_stride']
            self.cur_row += 1
            if _w1 + ssd['shape'][0] >= self.ms_resize_shape[0]:
                row = 0
                self.cur_row = 0
                col += img_cfg['slide_stride']
                self.cur_col += 1

    def cur_pos(self):
        return self.cur_row * img_cfg['slide_stride'] * self.crop_size[0] ,\
               self.cur_col * img_cfg['slide_stride'] * self.crop_size[1]


    def expand(self, l=0, t=0, b=0, r=0):
        x1 = self.cur_row * img_cfg['slide_stride'] * self.crop_size[0]
        x2 = x1 + ssd['shape'][0]
        y1 = self.cur_col * img_cfg['slide_stride'] * self.crop_size[1]
        y2 = y1 + ssd['shape'][1]

        return self.image[max(0, y1-t):min(y2+b, self.resize_shape[1]), max(x1-l, 0):min(x2+r, self.resize_shape[0]), :]



if __name__ == '__main__':
    image = Image(dataset_name='1-HIT_Canteen', frame_name='IMG_1_15')
    image.cv2_resize()
    image.cv2_toNumpy()
    image.normalize()
    image.hwc2chw()
