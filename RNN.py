import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=256, output_dim=None):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.representation_dim = hidden_dim

        if output_dim is not None:
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = None

    def forward(self, text, text_length):

        # [sentence len, batch size] => [sentence len, batch size, embedding size]
        embedded = self.embedding(text)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.cpu()).cuda()

        # [sentence len, batch size, embedding size] =>
        #  output: [sentence len, batch size, hidden size]
        #  hidden: [1, batch size, hidden size]
        packed_output, (hidden, cell) = self.rnn(packed)

        if self.fc is not None:
            return self.fc(hidden.squeeze(0)).view(-1)
        else:
            return hidden.squeeze(0)