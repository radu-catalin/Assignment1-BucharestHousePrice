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
import seaborn as sn
import pandas as pd



# device config
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
input_size = 6
hidden_size = 4000
output_size = 10
num_epochs = 500
batch_size = 1
learning_rate = 0.9
momentum = 0.6
log_interval = int(1000 / batch_size)


path = './data/Bucharest_HousePriceDataset.csv'


def get_dataset(path: str, shuffle: bool, interval: int, normalize: bool) -> DataLoader:
    np_dataset = np.genfromtxt(
        path,
        delimiter=',',
        usecols=(1, 2, 3, 4, 5, 6)
    ).astype(np.int32)[interval[0]:interval[1]]

    transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    if normalize == True:
        col_max = np_dataset.max(axis=0)
        # print(col_max)
        # exit()
        np_dataset = np_dataset / col_max

    np_labels = np.genfromtxt(
        path,
        delimiter=',',
        usecols=(0),
    ).astype(np.int32)[interval[0]:interval[1]]

    # print(np_labels)
    # print(np_labels.max())
    # exit()

    dataset, labels = map(
        torch.tensor,
        (np_dataset, np_labels)
    )
    dataset = transform(dataset)
    labels = transform(dataset)
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
    shuffle=True,
    interval=(3000, None),
    normalize=True
)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.linear1 = nn.Linear(
          in_features=input_size,
          out_features=hidden_size
        )
        self.linear2 = nn.Linear(
          in_features=hidden_size,
          out_features=output_size
        )

        # self.linear3 = nn.Linear(
        #     in_features=1500,
        #     out_features=output_size
        # )


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # x = self.linear3(x)
        # print(x.shape)
        # exit()
        return x


# model
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=learning_rate, amsgrad=True)
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
losses_train = []
# training loop
for epoch in range(num_epochs):
    model.train()
    l = 0
    iter = 0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device).float()
        labels = labels.to(device).long()

        outputs = model(data)
        # print(labels.shape, labels)
        # print(outputs.shape)
        # print('----')
        loss = criterion(outputs, labels)
        l += loss
        iter += 1
        # backward
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % log_interval == 0:
          print(
              f'epoch: {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}'
          )
    l /= iter
    losses_train.append(l)

print('Finished training!')

losses_validation = []
confusion_matrix = torch.zeros([10, 10], dtype=torch.int32)
classes = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to(device)
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device).float()
        labels = labels.to(device).long()

        outputs = model(data)

        loss = criterion(outputs, labels)
        losses_validation.append(loss)

        _, predicted = torch.max(outputs, dim=-1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(predicted)):
          confusion_matrix[labels[i].long(), predicted[i].long()] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc}%')

    # df_cm = pd.DataFrame(confusion_matrix.detach().cpu().numpy().astype(np.int32), index = [1, 2, 3, 4, 5, 6, 7 ,8, 9, 10], columns = [1, 2, 3, 4, 5, 6, 7 ,8, 9, 10])
    # plt.figure(figsize=(10,7))
    # sn.heatmap(df_cm, annot=True, fmt='g')
    # plt.xlabel('target')
    # plt.ylabel('predicted')
    # plt.show()

plt.figure(1)
plot_loss(losses_train, label='regression_loss', color='red')
plt.figure(2)
plot_loss(losses_validation, label='regression_validation_loss', color='blue')
plt.show()
