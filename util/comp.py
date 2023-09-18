import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, device, train_loader, validation_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []
    target_list, predicted_list = [], []
    with tqdm(range(epochs), unit='epoch') as tepochs:
        tepochs.set_description('Training')
        for epoch in tepochs:
            model.train()
            running_loss = 0
            correct, total = 0, 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepochs.set_postfix(loss=loss.item())
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            train_loss.append(running_loss / len(train_loader))
            train_acc.append(correct / total)

            model.eval()
            running_loss = 0
            correct, total = 0, 0
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                tepochs.set_postfix(loss=loss.item())
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                predicted_list.extend(predicted)
                target_list.extend(target)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            validation_loss.append(running_loss / len(validation_loader))
            validation_acc.append(correct / total)
    return train_loss, train_acc, validation_loss, validation_acc, predicted_list, target_list
