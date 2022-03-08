import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", default = 10, type = int)

parser.add_argument("--lr", default = 0.0001, type = float)

parser.add_argument("--model", default = "mobilenet_v2", type = str)

parser.add_argument("--train_data", default = "/mount_data/train_data", type = str)

parser.add_argument("--results_dir", default = "/mount_data/train_results", type = str)

args = parser.parse_args()

begin = time.time()
data_dir = args.train_data

print('loading data')

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize([224,224]),
                                       transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transforms = transforms.Compose([transforms.Resize([224,224]),
                                      transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader
trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

model = torch.hub.load('pytorch/vision:v0.5.0', args.model, pretrained=False, num_classes=2)
print(model)

criterion = nn.NLLLoss()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = args.lr)

print(type(trainloader))


epochs = args.epochs
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

os.makedirs(args.results_dir, exist_ok = True)

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        print(steps,'steps')
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = F.log_softmax(model(inputs), dim=1)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device),labels.to(device)
            logps = F.log_softmax(model(inputs), dim=1)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))                    

    print("Epoch: {}/{}".format((epoch+1),(epochs)),
                  "Train loss: {}".format(running_loss/len(trainloader)),
                  "Test loss:{}".format(test_loss/len(testloader)),
                  "Test accuracy:{}".format(accuracy/len(testloader)))
    running_loss = 0
    model.train()
    if epoch%2==0:
        torch.save(model.state_dict(), os.path.join(args.results_dir,'nist_fingermodel_bal_mv2_{}_{}_{}.pth'.format(args.lr, args.epochs, args.model)))
#torch.save(model, 'fingermodel.pth')


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig(os.path.join(args.results_dir,'results_{}_{}_{}.png'.format(args.lr, args.epochs, args.model)))

end = time.time()
t = end - begin
print("The time taken is: "+str(t))