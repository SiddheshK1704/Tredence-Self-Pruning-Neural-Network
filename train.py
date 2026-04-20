import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import PrunableNet
import config

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.BATCH_SIZE)

    return trainloader, testloader


def train_model(lambda_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PrunableNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader = get_data()

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)


            gates = model.get_all_gates()
            sparsity_loss = torch.sum(torch.abs(model.get_all_gates()))

            loss = cls_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Lambda {lambda_val}] Epoch {epoch+1}, Loss: {total_loss:.3f}")

    return model, testloader


def evaluate(model, testloader):
    device = next(model.parameters()).device
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, pred = outputs.max(1)

            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    accuracy = 100 * correct / total

    gates = model.get_all_gates().detach()
    sparsity = (gates < config.SPARSITY_THRESHOLD).float().mean().item() * 100

    return accuracy, sparsity, gates.cpu().numpy()