import os.path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# 2. Define a Convolutional Neural Network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(train=False):
    # 1. Load and normalize CIFAR10
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if not os.path.exists("./test_input.bin"):
        make_tune_testsets(trainloader, testloader)

    # 2. define the network
    net = Net()

    import torch.optim as optim

    # 3. Define a Loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    if train:
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        # save our trained model

        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)

    # 5. Test the network on the test data

    net = SeqNet()
    net.load_state_dict(torch.load("./cifar_net_seq.pth"))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(*([  # define both layers and how they work in forward function
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),  # Test if you need inplace relu
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        ]))
        self.linears = nn.Sequential(*([
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        ]))
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.convolutions(x)
        x = self.linears(x)
        return x
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

def make_tune_testsets(trainloader, testloader):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("making lists of data")
    tune = list(trainloader)
    test = list(testloader)
    tunearr = torch.zeros((10000, 3, 32, 32))
    tunelabels = torch.zeros((10000,))
    testarr = torch.zeros((10000, 3, 32, 32))
    testlabels = torch.zeros((10000,))
    for i, (tune, test) in enumerate(zip(tune, test)):
        for j in range(4):
            tunearr[i * j + i] = tune[0][j]
            tunelabels[i*j + i] = tune[1][j]
            testarr[i * j + i] = test[0][j]
            testlabels[i*j + i] = test[1][j]
        if i * j + i == 9999:
            break
    print("writing files")
    convert_dataset_to_binary(tunearr.numpy(), tunelabels.numpy(), testarr.numpy(), testlabels.numpy())


def convert_network_to_sequential():
    PATH = './cifar_net.pth'

    net = Net()
    net.load_state_dict(torch.load(PATH))
    layers = list(net.children())
    new_net = SeqNet()
    new_layers = list(new_net.children())
    print("\n".join([str(x) for x in layers]),"\nNew layers:\n", new_layers)
    #Manually assign new layers into the sequential, so that we transfer the weights into the sequential model
    new_layers[0][0] = layers[0]
    new_layers[0][3] = layers[2]
    new_layers[1][0] = layers[3]
    new_layers[1][2] = layers[4]
    new_layers[1][4] = layers[5]


    torch.save(new_net.state_dict(), "./cifar_net_seq.pth")

def convert_dataset_to_binary(tune, tunelabels, test, testlabels):

    with open(f"tune_input.bin", "wb") as fd:
        tune.astype("<f4").tofile(fd)  # Convert to little endian and save.

    with open(f"tune_labels.bin", "wb") as fd:
        tunelabels.astype("int32").tofile(fd)

    with open(f"test_input.bin", "wb") as fd:
        test.astype("<f4").tofile(fd)  # Convert to little endian and save.

    with open(f"test_labels.bin", "wb") as fd:
        testlabels.astype("int32").tofile(fd)


if __name__ == '__main__':
    #convert_network_to_sequential()
    #quit()
    main()
