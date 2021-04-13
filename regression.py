import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms


# device config
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
input_size = 6
hidden_size = 10
output_size = 1
num_epochs = 10
batch_size = 5
learning_rate = 0.01
momentum = 0.9
log_interval = int(1000 / batch_size)


path = './data/Bucharest_HousePriceDataset.csv'


def get_dataset(path: str, shuffle: bool, interval: int, normalize: bool) -> DataLoader:
    np_dataset = np.genfromtxt(
        path,
        delimiter=',',
        usecols=(1, 2, 3, 4, 5, 6)
    ).astype(np.int32)[interval[0]:interval[1]]

    if normalize == True:
        col_max = np_dataset.max(axis=0)
        np_dataset = np_dataset / col_max

    np_labels = np.genfromtxt(
        path,
        delimiter=',',
        usecols=(0),
    ).astype(np.int32)[interval[0]:interval[1]]

    dataset, labels = map(
        torch.tensor,
        (np_dataset, np_labels)
    )

    dataset = dataset.to(device)
    labels = labels.to(device)

    dataset = TensorDataset(dataset, labels)
    dataset_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataset_loader


def plot_loss(loss, label, color='blue'):
    plt.plot(loss, label=label, color=color)
    plt.legend()


train_loader = get_dataset(
    path=path,
    shuffle=True,
    interval=(1, 3000),
    normalize=True
)

test_loader = get_dataset(
    path=path,
    shuffle=False,
    interval=(3000, None),
    normalize=True
)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.linear1 = nn.Linear(in_features=input_size,
                                 out_features=hidden_size)
        self.linear2 = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# model
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum)

n_total_steps = len(train_loader)
losses_train = []
# training loop
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device).float()
        # print(data)
        labels = labels.to(device).view(-1, 1).float()

        outputs = model(data)

        loss = criterion(outputs, labels)

        # backward
        losses_train.append(loss.detach().cpu().numpy())
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % log_interval == 0:
            print(
                f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
            )

print('Finished training!')

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device).float()
        labels = labels.to(device).view(-1, 1).float()

        outputs = model(data)

        n_samples += labels.size(0)
        n_correct += (outputs.int() + 1 == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc}%')
plot_loss(losses_train, label='loss')
plt.show()
