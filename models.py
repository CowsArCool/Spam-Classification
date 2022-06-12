import torch
import torch.nn as nn


class basic_LSTM (nn.Module):
    def __init__(self,
                 args,
                 input_dim,
                 output_dim):

        super(basic_LSTM, self).__init__()

        self.hidden_dim = args.hidden_dim

        self.embedding = nn.Embedding(input_dim, args.embedding_dim)

        self.lstm = nn.LSTM(
            args.embedding_dim,
            args.hidden_dim,
            num_layers=args.num_layers)

        self.fc = nn.Linear(args.hidden_dim, output_dim)

        self.name = 'basic_LSTM'

    def forward(self, x):
        batch_size = x.size(1)

        embed = self.embedding(x)

        lstm_output, (lstm_hidden, cell) = self.lstm(embed)

        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)

        # print(f'LSTM Out Shape {lstm_output.shape}')

        out = self.fc(lstm_output)

        # print(f'FC out shape {out.shape}')

        out = out.view(batch_size, -1)

        # print(f'View out shape {out.shape}')

        out = out[:, -1]

        # print(out)
        # print(f'Out Shape {out.shape}\tOut Mean {torch.mean(out)}\n')
        return out


def basic_RNN():
    pass
