from Models import *
from Models.CRNN import CRNN

rcnn = CRNN()
rcnn.load()

actions = ['Kayaking']
all_names = ['Kayaking.mp4']

prediction = rcnn.run('/home/emad/PycharmProjects/video_classification_sample/video', all_names, actions)
print(prediction)