import pickle
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


class Network(torch.nn.Module):
    def __init__(self, input_size, layer_size):
        super(Network, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, layer_size)
        self.layer2 = torch.nn.Linear(layer_size, layer_size)
        self.layer3 = torch.nn.Linear(layer_size, 9)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def train(epoch, model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch : {epoch}\tLoss : {loss}")


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"\nTest set accuracy : [{correct} / {len(test_loader.dataset)}] ({(100 * correct / len(test_loader.dataset)):.0f}%)\n")


if __name__ == "__main__":
    # HYPERPARAMETER
    learning_rate = 0.001
    batch_size = 64
    epoch_size = 10

    with open('data/X_MFE.pickle', 'rb') as f:
        train_x = pickle.load(f)
        train_x = torch.tensor(train_x, dtype=torch.float64)

    with open('data/y.pickle', 'rb') as f:
        train_y = pickle.load(f)
        train_y = torch.tensor(train_y, dtype=torch.long)

    dataset = TensorDataset(train_x, train_y)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Network(input_size=59, layer_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epoch_size):
        train(epoch=epoch, model=model, train_loader=train_loader, optimizer=optimizer)
        test(model=model, test_loader=test_loader)
