import cv2
import time
import subprocess as sp
import multiprocessing
import psutil
from oldcare.utils import streampushassistant
from oldcare.utils.pathassistant import get_path


class stream_pusher(object):
    def __init__(self, rtmp_url=None, raw_frame_q=None):
        self.rtmp_url = rtmp_url
        self.raw_frame_q = raw_frame_q

        fps = 20
        width = 640
        height = 480

        self.command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(width, height),
                        '-r', str(fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        self.rtmp_url]

    def __frame_handle__(self, raw_frame, text, shape1, shape2):
        return (raw_frame)

    def push_frame(self):
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
        p = sp.Popen(self.command, stdin=sp.PIPE)

        while True:
            if not self.raw_frame_q.empty():
                raw_frame, text, shape1, shape2 = self.raw_frame_q.get()
                frame = self.__frame_handle__(raw_frame, text, shape1, shape2)

                p.stdin.write(frame.tostring())
            else:
                time.sleep(0.01)

    def run(self):
        push_frame_p = multiprocessing.Process(target=self.push_frame, args=())
        push_frame_p.daemon = True
        push_frame_p.start()


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    rtmpUrl = 'rtmp://182.92.84.33:1935/stream/base'
    raw_q = multiprocessing.Queue()

    my_pusher = stream_pusher(rtmp_url=rtmpUrl, raw_frame_q=raw_q)
    my_pusher.run()
    while True:
        _, raw_frame = cap.read()
        info = (raw_frame, '2', '3', '4')
        if not raw_q.full():
            raw_q.put(info)
        cv2.waitKey(1)
    cap.release()

    # pusher = streampushassistant.get_pusher(rtmpUrl)
    # while True:
    #     _, raw_frame = cap.read()
    #     streampushassistant.push(pusher, raw_frame)
    #     cv2.waitKey(1)
    # cap.release()