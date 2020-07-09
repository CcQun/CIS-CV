import cv2

rtmpurl = 'rtmp://182.92.84.33:1935/stream/base'
cam = cv2.VideoCapture(rtmpurl)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

while True:
    _, image = cam.read()
    print(type(image))
    cv2.imshow('Stream pulling', image)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break