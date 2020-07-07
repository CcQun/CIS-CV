# -*- coding: utf-8 -*-
'''
义工是否和老人有互动主程序

用法：
python checkingvolunteeractivity.py
python checkingvolunteeractivity.py --filename tests/desk_01.mp4
'''

from oldcare.facial import FaceUtil
from oldcare.utils.insertingassistant import inserting
from oldcare.utils.pathassistant import get_path
from oldcare.utils import communicationassistant
from scipy.spatial import distance as dist
from oldcare.utils import fileassistant
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import os
import imutils
import numpy as np
import argparse
import subprocess

# 更新一下people_info.csv
communicationassistant.get_people_info()

activity_type = 4

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=False, default='',
                help="")
args = vars(ap.parse_args())

# 全局变量
pixel_per_metric = None
input_video = args['filename']
output_activity_path = get_path('output_activity_path', 1)
model_path = get_path('facial_recognition_model_path')
people_info_path = get_path('people_info_path')
camera_turned = 0

# 全局常量
FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20
ACTUAL_DISTANCE_LIMIT = 100  # cm

# 得到 ID->姓名的map 、 ID->职位类型的map
id_card_to_name, id_card_to_type = fileassistant.get_people_info(
    people_info_path)

# 初始化摄像头
if not input_video:
    vs = cv2.VideoCapture(0)
    time.sleep(2)
else:
    vs = cv2.VideoCapture(input_video)

# 加载模型
faceutil = FaceUtil(model_path)

print('[INFO] 开始检测义工和老人是否有互动...')
# 不断循环
counter = 0
while True:
    counter += 1
    camera_turned = 0
    # grab the current frame
    (grabbed, frame) = vs.read()

    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if input_video and not grabbed:
        break

    if not input_video:
        frame = cv2.flip(frame, 1)

    frame = imutils.resize(frame,
                           width=VIDEO_WIDTH,
                           height=VIDEO_HEIGHT)  # 压缩，加快识别速度

    face_location_list, names = faceutil.get_face_location_and_name(
        frame)

    # 得到画面的四分之一位置和四分之三位置，并垂直划线
    one_sixth_image_center = (int(VIDEO_WIDTH / 6), int(VIDEO_HEIGHT / 6))
    five_sixth_image_center = (int(VIDEO_WIDTH / 6 * 5),
                               int(VIDEO_HEIGHT / 6 * 5))

    cv2.line(frame, (one_sixth_image_center[0], 0),
             (one_sixth_image_center[0], VIDEO_HEIGHT),
             (0, 255, 255), 1)
    cv2.line(frame, (five_sixth_image_center[0], 0),
             (five_sixth_image_center[0], VIDEO_HEIGHT),
             (0, 255, 255), 1)

    people_type_list = list(set([id_card_to_type[i] for i in names]))

    volunteer_name_direction_dict = {}
    volunteer_centroids = []
    old_people_centroids = []
    old_people_name = []

    # loop over the face bounding boxes
    for ((left, top, right, bottom), name) in zip(face_location_list,
                                                  names):  # 处理单个人

        person_type = id_card_to_type[name]
        # 将人脸框出来
        rectangle_color = (0, 0, 255)
        if person_type == 'old_people':
            rectangle_color = (0, 0, 128)
        elif person_type == 'employee':
            rectangle_color = (255, 0, 0)
        elif person_type == 'volunteer':
            rectangle_color = (0, 255, 0)
        else:
            pass
        cv2.rectangle(frame, (left, top), (right, bottom),
                      rectangle_color, 2)

        if 'volunteer' not in people_type_list:  # 如果无义工则跳出循环
            continue

        if person_type == 'volunteer':  # 如果检测到有义工存在
            # 获得义工位置
            volunteer_face_center = (int((right + left) / 2),
                                     int((top + bottom) / 2))
            volunteer_centroids.append(volunteer_face_center)

            cv2.circle(frame,
                       (volunteer_face_center[0],
                        volunteer_face_center[1]),
                       8, (255, 0, 0), -1)

            adjust_direction = ''
            # face locates too left, servo need to turn right,
            # so that face turn right as well
            if volunteer_face_center[0] < one_sixth_image_center[0]:
                adjust_direction = 'right'
            elif volunteer_face_center[0] > five_sixth_image_center[0]:
                adjust_direction = 'left'

            volunteer_name_direction_dict[name] = adjust_direction

        elif person_type == 'old_people':  # 如果没有发现义工
            old_people_face_center = (int((right + left) / 2),
                                      int((top + bottom) / 2))
            old_people_centroids.append(old_people_face_center)
            old_people_name.append(name)

            cv2.circle(frame,
                       (old_people_face_center[0],
                        old_people_face_center[1]),
                       4, (0, 255, 0), -1)
        else:
            pass

        # 把人名写上 (同时处理中文显示问题)
        img_PIL = Image.fromarray(cv2.cvtColor(frame,
                                               cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        final_label = id_card_to_name[name]
        draw.text((left, top - 30), final_label,
                  font=ImageFont.truetype('simsun.ttc',
                                          40), fill=(255, 0, 0))  # linux
        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

    # 义工追踪逻辑
    if 'volunteer' in people_type_list:
        volunteer_adjust_direction_list = list(
            volunteer_name_direction_dict.values())
        # 有的义工恰好在范围内，所以不需要调整舵机
        if '' in volunteer_adjust_direction_list:
            print('%d-有义工恰好在可见范围内，摄像头不需要转动'
                  % (counter))
        else:
            adjust_direction = volunteer_adjust_direction_list[0]
            camera_turned = 1
            print('%d-摄像头需要 turn %s %d 度' % (counter,
                                             adjust_direction, ANGLE))

    # 在义工和老人之间划线
    if camera_turned == 0:
        for i in volunteer_centroids:
            for j_index, j in enumerate(old_people_centroids):
                pixel_distance = dist.euclidean(i, j)
                face_pixel_width = sum([i[2] - i[0] for i
                                        in face_location_list]) / len(face_location_list)
                pixel_per_metric = face_pixel_width / FACE_ACTUAL_WIDTH
                actual_distance = pixel_distance / pixel_per_metric

                if actual_distance < ACTUAL_DISTANCE_LIMIT:
                    cv2.line(frame, (int(i[0]), int(i[1])),
                             (int(j[0]), int(j[1])), (255, 0, 255), 2)
                    label = 'distance: %dcm' % (actual_distance)
                    cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    event_desc = '%s正在与义工交互' % (id_card_to_name[old_people_name[j_index]])
                    event_location = '房间桌子'
                    print('[EVENT] %s, 房间桌子, %s 正在与义工交互.'
                          % (current_time,
                             id_card_to_name[old_people_name[j_index]]))

                    # event_desc, event_type, event_location, old_people_id, output_path, frame
                    inserting(event_desc, activity_type, event_location, int(name), output_activity_path,
                              frame)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Checking Volunteer's Activities", frame)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()
