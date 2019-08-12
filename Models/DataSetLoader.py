import numpy as np
import torch
from torch.utils import data
import cv2
from PIL import Image


class DataSetLoader(data.Dataset):
    def __init__(self, data_path, file_names, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.file_names = file_names
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.file_names)

    def read_images(self, path, selected_folder, use_transform):
        X = []

        vidcap = cv2.VideoCapture(path + "/" + selected_folder)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
            trans_img = np.copy(image)
            if success and use_transform is not None:
                PIL_image = Image.fromarray(image)
                trans_img = use_transform(PIL_image)

            if success and count <= len(self.frames):
                X.append(trans_img)
            else:
                break

        return torch.stack(X, dim=0)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        file = self.file_names[index]

        # Load data
        X = self.read_images(self.data_path, file, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y