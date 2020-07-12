import os
import playsound

from oldcare.utils.pathassistant import get_path

playsound.playsound(os.path.join(get_path('audio_path'), 'blink.mp3'))
