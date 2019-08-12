import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from torch.utils import data

from Models import CNNEncoder, RNNDecoder
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms

from Models.DataSetLoader import DataSetLoader


class CRNN():

    def __init__(self):
        # use same encoder CNN saved!
        self.CNN_fc_hidden1, self.CNN_fc_hidden2 = 1024, 768
        self.CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        self.res_size = 224  # ResNet image size
        self.dropout_p = 0.0  # dropout probability

        self.RNN_hidden_layers = 3
        self.RNN_hidden_nodes = 512
        self.RNN_FC_dim = 256

        self.k = 101  # number of target category
        self.batch_size = 40
        self.begin_frame, self.end_frame, self.skip_frame = 1, 29, 1

        self.label_encoder = LabelEncoder()
        self.action_category = []

        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")

        self.encoder = CNNEncoder.CNNEncoder(self.CNN_fc_hidden1, self.CNN_fc_hidden2, self.dropout_p,
                                             self.CNN_embed_dim).to(device)

        self.decoder = RNNDecoder.RNNDecoder(self.CNN_embed_dim, self.RNN_hidden_layers, self.RNN_hidden_nodes,
                                             self.RNN_FC_dim, self.dropout_p, self.k).to(device)

    def load(self):
        save_model_path = '/home/emad/PycharmProjects/video_classification_sample/cached'
        self.encoder.load_state_dict(
            torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch63_singleGPU.pth'), map_location='cpu'))
        self.decoder.load_state_dict(
            torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch63_singleGPU.pth'), map_location='cpu'))

        with open(save_model_path + '/UCF101actions.pkl', 'rb') as f:
            action_names = pickle.load(f)  # load UCF101 actions names

        # convert labels -> category
        self.label_encoder.fit(action_names)
        self.action_category = self.label_encoder.transform(action_names).reshape(-1, 1)

    def run(self, path, x, y):
        # data loading parameters
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
        params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 4,
                  'pin_memory': True} if use_cuda else {}

        transform = transforms.Compose([transforms.Resize([self.res_size, self.res_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        selected_frames = np.arange(self.begin_frame, self.end_frame, self.skip_frame).tolist()

        # reset data loader
        all_data_params = {'batch_size': self.batch_size, 'shuffle': False, 'num_workers': 4,
                           'pin_memory': True} if use_cuda else {}

        y = self.label_encoder.transform(y)

        data_loader = data.DataLoader(
            DataSetLoader(path, x, y, selected_frames, transform=transform), **all_data_params)

        self.encoder.eval()
        self.decoder.eval()

        all_y_pred = []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(tqdm(data_loader)):
                # distribute data to device
                X = X.to(device)
                output = self.decoder(self.encoder(X))
                y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
                all_y_pred.append(y_pred.cpu().data.squeeze().numpy().tolist())

        return self.label_encoder.inverse_transform(all_y_pred)
