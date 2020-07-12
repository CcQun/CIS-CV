# -*- coding: utf-8 -*-
'''
audio player
'''

# import library
# from subprocess import call
import playsound



# play audio
# def play_audio(audio_name):
#     try:
#         call('mpg321 ' + audio_name, shell=True)  # use mpg321 player
#     except KeyboardInterrupt as e:
#         print(e)
#     finally:
#         pass


def play_audio(audio_name):
    playsound.playsound(audio_name)


if __name__ == '__main__':
    pass
