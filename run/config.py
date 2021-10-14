"""
GIGA十亿像素检测系统
配置文件
配置说明：
ROOT：项目根目录，即GIGA所在的位置

"""
ROOT = ''                       #根目录位置
GPU_ID = '0'                    #GPU选择
#detection 检测功能配置

PRETRAIN = False                #是否使用预训练模型
MODEL_NAME = 'faster_rcnn'      #可选faster_rcnn cascade_rcnn retina_net
DRAW_RESULT = True              #是否可视化结果
#train
TRAIN_MODEL_NAME = 'faster_rcnn'#可选faster_rcnn cascade_rcnn retina_net

