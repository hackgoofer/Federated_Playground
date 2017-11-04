from __future__ import print_function
import argparse
import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from copy import deepcopy

FEDERATED = 0
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("Processing data in memory...")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   batch_size=args.test_batch_size, shuffle=True, **kwargs)

n = 2
total_train_size = len(train_loader)
child_data_size = (int)(total_train_size/n)

child1_train= train_loader.collate_fn([train_loader.dataset[i] for i in range(0, child_data_size)])
child2_train= train_loader.collate_fn([train_loader.dataset[i] for i in range(child_data_size+1, total_train_size)])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(epoch, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

# this applies model on each child
def train_federated(epoch, child_train_data, model, optimizer):
    model.train()
    batch_size = args.batch_size
    print("Batch_size: " + str(batch_size))
    num_batches = math.floor(len(child_train_data[0])/batch_size)
    print("Num_batches: " + str(num_batches))
    final_gradients = []

    nounce = 0
    for batch_idx in range(0, num_batches):
        data = child_train_data[0][nounce:nounce+batch_size]
        target = child_train_data[1][nounce:nounce+batch_size]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        parameters = list(model.parameters())
        gradients = list(map(lambda x: x.grad, parameters))
        if len(final_gradients) == 0:
            final_gradients = gradients
        else:
            for p1, p2 in zip(final_gradients, gradients):
                p1.add_(p2)
        if 0:
            import pdb; pdb.set_trace()
        optimizer.step()
        nounce = batch_size + nounce
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(child_train_data[1]),
                100. * batch_idx / len(child_train_data[1]), loss.data[0]))

    for i in range(0, len(final_gradients)):
        final_gradients[i] = final_gradients[i]/num_batches
    return final_gradients

def test(model):
    model.eval()
    correct = 0
    test_loss = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def federated_learning():
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model1 = Net()
    model2 = Net()
    if args.cuda:
        model = model.cuda()
        model1 = model1.cuda()
        model2 = model2.cuda()

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        grad1 = train_federated(epoch, child1_train, model1, optimizer1)
        grad2 = train_federated(epoch, child2_train, model2, optimizer2)

        # calculate the average grad
        grad = deepcopy(grad1)
        for a, b in zip(grad, grad2):
            a.add_(b)

        for i in range(0, len(grad)):
            grad[i] = grad[i]/2

        # update the master model
        if len(optimizer.state_dict()['state']) == 0:
            updateOptimizer(optimizer, grad, optimizer1)
        else:
            updateOptimizer(optimizer, grad, None)
        optimizer.step()

        # propagate the updated model back to the child
        updateChildModelFromParent(model1, model)
        updateChildModelFromParent(model2, model)

        test(model)
        print("epoch: " + str(epoch))

def updateChildModelFromParent(child, parent):
    for p_parent, p_child in zip(parent.parameters(), child.parameters()):
        p_child.data.copy_(p_parent.data)

def updateOptimizer(op, grad, copy_op):
    op_state = op.state_dict()['state']
    op_param = op.state_dict()['param_groups']
    if copy_op:
        for key, value in copy_op.state_dict()['state'].items():
            op_state[key] = deepcopy(value)
    i = 0
    for key, value in op_state.items():
        op_state[key] = deepcopy(grad[i])
        i = i + 1
    if 0:
        import pdb; pdb.set_trace()
#    op.load_state_dict({'state': op_state, 'param_groups': op_param})
    op.param_groups[0]['params'][0].grad = deepcopy(grad)

def updateOptimizer1(op, grad, copy_op):
    for group in op.param_groups:
        for p in group['params']:
            if copy_op:
                p.grad = deepcopy(grad)
            else:
                p.grad.data.copy_(grad)


def updateStateWithGradients(state, grad):
    i = 0
    new_state = {}
    for name, weight in state.items():
        weight.add_(-args.lr, grad[i].data)
        i = i + 1
        new_state[name] = weight
    return new_state

if FEDERATED:
    model = Net()
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer)
        test(model)
else:
    federated_learning()


