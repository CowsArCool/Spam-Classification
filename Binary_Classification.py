# Base Packages
import os
import numpy as np
import pandas as pd
import wandb

# Preprocessing
from Preprocessing import SpamDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Torch
import torch
import torch.nn as nn

# Models
from models import basic_LSTM

# Random Crap
from alive_progress import alive_bar, config_handler

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(
    dir_path,
    "ham-spam",
    "spam.csv"
))

df.drop(df.columns[[np.arange(2, 5)]], axis=1, inplace=True)
df.rename(columns={'v1': 'labels', 'v2': 'text'}, inplace=True)
df.head(n=6)

train, test = train_test_split(df, test_size=0.2)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')


test_dataset = SpamDataset(train)
# print (len (test_dataset.stoi))

trainset = SpamDataset(train)
testset = SpamDataset(test)

trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

print('Datasets Built')

# print (list (trainset.stoi.items())[:30])


input_dim = len(trainset.stoi)
print(f'input dim: {input_dim}')
print(f'len trainset: {len(trainset)}')

embedding_dim = 100
hidden_dim = 256
output_dim = 1
lr = 1e-6
num_epochs = 100

use_gpu = ('cuda' if torch.cuda.is_available else 'cpu')
print(f'Using Device: {use_gpu}')
device = torch.device(use_gpu)

model = basic_LSTM(
    input_dim,
    output_dim
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()


config = {
    'lr': lr,
    'hidden dim': hidden_dim,
    'embedding_dim': 100
}

wandb.init(
    project='TextSpamClassification',
    entity='cowsarecool',
    config=config
)

wandb.watch(model)


for epoch in range(num_epochs):
    epoch_losses = []
    epoch_acc = []

    model.train()
    with alive_bar(len(trainloader),
                   title='Training', bar='smooth',
                   length=75) as bar:
        for batch_num, batch in enumerate(trainloader):
            src = (batch['src']
                   .to(device)
                   .transpose(0, 1)
                   )

            trg = batch['trg'].type(torch.float).to(device)

            optimizer.zero_grad()

            output = model(src)

            # assert output.shape[0] == trg.shape[0]

            loss = criterion(output, trg)
            epoch_losses.append(loss.item())

            wandb.log({
                'loss': loss,
                'epoch': epoch
            })

            rounded_pred = torch.round(torch.sigmoid(output))  # WTF
            correct = (rounded_pred == trg).float()

            acc = correct.sum()/len(correct)
            epoch_acc.append(acc)

            loss.backward()

            optimizer.step()

            bar.text(f'Epoch Step: {batch_num+1}')
            bar()

    model.eval()

    test_losses = []
    test_accuracies = []
    # with torch.nograd
    for batch in testloader:
        test_src = batch['src'].to(device)
        test_src = test_src.transpose(0, 1)

        test_trg = batch['trg'].type(torch.float).to(device)

        output = model(test_src)
        # print('Output Shape {}'.format(output.shape))
        # print('Target Shape: {}'.format(test_trg.shape))

        test_loss = criterion(output, test_trg)
        test_accuracy = (
            torch.sum(
                torch.round(
                    torch.sigmoid(output)
                ) == test_trg)
        )/len(test_trg)

        test_losses.append(test_loss.item())
        test_accuracies.append(test_accuracy.item())

        # print(output)

    wandb.log({
        'test accuracy': np.mean(test_accuracies),
        'test loss': np.mean(test_losses),
        'epoch': epoch,
    })

    print(
        f'\nEpoch: {epoch}\nAvg Loss: {np.round(np.mean(epoch_losses), decimals=3)}\
          \tTest Loss: {np.round(np.mean(test_losses), decimals=3)}\
          \tTest Accuracy: {np.round((np.mean(test_accuracies)*100), decimals=3)}%\
          ')

wandb.finish()
