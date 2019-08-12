import os

from Models import *
from Models.CRNN import CRNN

rcnn = CRNN()
rcnn.load()

actions = ['Kayaking']
all_names = ['Kayaking.mp4']
project_path = os.getcwd()

prediction = rcnn.run(project_path + '/video', all_names, actions)
print(prediction)
