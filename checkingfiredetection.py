# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

用法：
python checkingfiredetection.py
python checkingfiredetection.py --filename tests/corridor_01.avi
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
    my_pusher = stream_pusher(rtmp_url=get_path('rtmp_fire_output', 2), raw_frame_q=raw_q)
    my_pusher.run()

    fire_type = 6

    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, default='',
                    help="")
    args = vars(ap.parse_args())
    input_video = args['filename']

    # 控制陌生人检测
    fire_timing = 0  # 计时开始
    fire_start_time = 0  # 开始时间
    fire_limit_time = 1

    # 全局变量
    model_path = get_path('fire_model_path')
    output_fire_path = get_path('output_fire_path', 1)

    # 全局常量
    TARGET_WIDTH = 48
    TARGET_HEIGHT = 48

    # 初始化摄像头
    if not input_video:
        vs = cv2.VideoCapture(get_path('rtmp_fire_input', 2))
        time.sleep(2)
    else:
        vs = cv2.VideoCapture(input_video)

    # 加载模型
    model = load_model(model_path)

    print('[INFO] 开始检测是否发生火灾...')
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
        (normal, fire) = model.predict(roi)[0]
        label = "Fire (%.2f)" % (fire) if fire > normal else "Normal (%.2f)" % (normal)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (image.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if fire > normal:
            if fire_timing == 0:  # just start timing
                fire_timing = 1
                fire_start_time = time.time()
            else:  # alredy started timing
                fire_end_time = time.time()
                difference = fire_end_time - fire_start_time

                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))

                if difference < fire_limit_time:
                    print('[INFO] %s, 房间, 火焰仅出现 %.1f 秒. 忽略.'
                          % (current_time, difference))
                else:  # strangers appear
                    event_desc = '有火灾发生!!!'
                    event_location = '房间'
                    print('[EVENT] %s, 房间, 有火灾发生!!!' % (current_time))

                    # event_desc, event_type, event_location, old_people_id, output_path, frame
                    inserting(event_desc, fire_type, event_location, None, output_fire_path,
                              image)

        # cv2.imshow('Fire detection', image)

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
