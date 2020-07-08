# -*- coding: utf-8 -*-

'''
训练人脸识别模型
'''

# import the necessary packages
import sys
from imutils import paths
from oldcare.facial import FaceUtil

# global variable
from oldcare.utils import communicationassistant
from oldcare.utils.pathassistant import get_path

userId = sys.argv[1]

dataset_path = get_path('imagedir')
output_encoding_file_path = get_path('facial_recognition_model_path')

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
image_paths = list(paths.list_images(dataset_path))

if len(image_paths) == 0:
    print('[ERROR] no images to train.')
else:
    faceutil = FaceUtil()
    print("[INFO] training face embeddings...")
    faceutil.save_embeddings(image_paths, output_encoding_file_path, userId)
    url = "http://localhost:10000/else/trainfrfeedback"
    request = {'userId': userId,
               'message': '训练完成'}
    response = communicationassistant.get_response(url, request)
    if response == 'error':
        print('[ERROR] 发送失败')
    count = 0
    exit(0)