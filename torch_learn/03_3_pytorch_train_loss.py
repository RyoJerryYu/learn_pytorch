from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

'''
Code NOT working'''

model = nn.Module()
train_loader = DataLoader()
val_loader = DataLoader()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output: torch.Tensor = model(data)
        loss: torch.Tensor = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))


def val(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss / len(val_loader.dataset)
    print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, val_loss))
