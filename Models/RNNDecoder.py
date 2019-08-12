import torch.nn as nn
import torch.nn.functional as f


class RNNDecoder(nn.Module):

    def __init__(self, cnn_embed_dim=300, h_rnn_layers=3, h_rnn=256, h_fc_dim=128, drop_p=0.3, num_classes=50):
        super(RNNDecoder, self).__init__()

        self.cnn_embed_dim = cnn_embed_dim
        self.h_rnn_layers = h_rnn_layers  # RNN hidden layers
        self.h_rnn = h_rnn  # RNN hidden nodes
        self.h_fc_dim = h_fc_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=cnn_embed_dim,
            hidden_size=h_rnn,
            num_layers=h_rnn_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_rnn, self.h_fc_dim)
        self.fc2 = nn.Linear(self.h_fc_dim, self.num_classes)

    def forward(self, x_rnn):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_rnn, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = f.relu(x)
        x = f.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
