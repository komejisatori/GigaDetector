# generate frames from video
import os
import cv2
from sys import path
path.append(r'../')
from data.video.config import video

def Video_2frame(src_name=video['video_name']):
    """
    generate frames for given video
    :param src_name: video name
    :return:
    """
    EXPORT_PATH = './newdataset'
    video_path = os.path.join(video['dataset_path'], src_name)
    frame_interval = video['frame_interval']
    store_name = video['video_name'][:-4]
    frame_path = EXPORT_PATH
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        success = True
    else:
        success = False
        print("[video error] open file failed!")

    frame_index = 0
    frame_count = 0
    while(success):
        success, frame = cap.read()
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        if frame_index % frame_interval == 0:
            cv2.imwrite(frame_path + "/%d.jpg" % frame_count, frame)
            frame_count += 1
            print("[video log] read %d frame" % frame_count)

        frame_index += 1

    cap.release()


if __name__ == "__main__":
    Video_2frame(src_name='The Balloon and the Wind.mp4')
