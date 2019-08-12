import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as f


class CNNEncoder(nn.Module):
    def __init__(self, fc_hidden1, fc_hidden2, drop_p, cnn_embed_dim=300):
        super(CNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, cnn_embed_dim)

    def forward(self, x_3d_tensor):
        cnn_embed = []
        for t in range(x_3d_tensor.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d_tensor[:, t, :, :, :])
                x = x.view(x.size(0), -1)

            x = self.bn1(self.fc1(x))
            x = f.relu(x)
            x = self.bn2(self.fc2(x))
            x = f.relu(x)
            x = f.dropout(x, self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq