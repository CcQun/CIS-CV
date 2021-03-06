# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

用法：
python checkingfalldetection.py
python checkingfalldetection.py --filename tests/corridor_01.avi
'''

# import the necessary packages
import multiprocessing

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from oldcare.utils.insertingassistant import inserting
from oldcare.utils.pathassistant import get_path
import numpy as np
import cv2
import time
import argparse

from oldcare.utils.streampushassistant import stream_pusher

if __name__ == '__main__':
    raw_q = multiprocessing.Queue()
    my_pusher = stream_pusher(rtmp_url=get_path('rtmp_corridor_output', 2), raw_frame_q=raw_q)
    my_pusher.run()

    fall_type = 2

    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, default='',
                    help="")
    args = vars(ap.parse_args())
    input_video = args['filename']

    # 控制陌生人检测
    fall_timing = 0  # 计时开始
    fall_start_time = 0  # 开始时间
    fall_limit_time = 1  # if >= 1 seconds, then he/she falls.

    # 全局变量
    model_path = get_path('fall_model_path')
    output_fall_path = get_path('output_fall_path', 1)

    # 全局常量
    TARGET_WIDTH = 64
    TARGET_HEIGHT = 64

    # 初始化摄像头
    if not input_video:
        vs = cv2.VideoCapture(get_path('rtmp_corridor_input', 2))
        time.sleep(2)
    else:
        vs = cv2.VideoCapture(input_video)

    # 加载模型
    model = load_model(model_path)

    print('[INFO] 开始检测是否有人摔倒...')
    # 不断循环
    counter = 0
    while True:
        counter += 1
        # grab the current frame
        (grabbed, image) = vs.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if input_video and not grabbed:
            break

        if not input_video:
            image = cv2.flip(image, 1)

        roi = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine facial expression
        (normal, fall) = model.predict(roi)[0]
        label = "Fall (%.2f)" % (fall) if fall > normal else "Normal (%.2f)" % (normal)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (image.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if fall > normal:
            if fall_timing == 0:  # just start timing
                fall_timing = 1
                fall_start_time = time.time()
            else:  # alredy started timing
                fall_end_time = time.time()
                difference = fall_end_time - fall_start_time

                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))

                if difference < fall_limit_time:
                    print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.'
                          % (current_time, difference))
                else:  # strangers appear
                    event_desc = '有人摔倒!!!'
                    event_location = '走廊'
                    print('[EVENT] %s, 走廊, 有人摔倒!!!' % (current_time))

                    # event_desc, event_type, event_location, old_people_id, output_path, frame
                    inserting(event_desc, fall_type, event_location, None, output_fall_path,
                              image)

        # cv2.imshow('Fall detection', image)

        info = (image, '2', '3', '4')
        if not raw_q.full():
            raw_q.put(info)
        cv2.waitKey(1)

        # Press 'ESC' for exiting videodat
        # GetFaceData.py
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vs.release()
    cv2.destroyAllWindows()
