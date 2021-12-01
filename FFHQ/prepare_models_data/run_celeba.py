# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from matplotlib import pyplot as plt
import os
import glob
import sys
sys.path.append('../')

from zipfile import ZipFile

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from celeba_data import celeba
from metrics.img_classifier import MobileNet


def train(model, epochs, train_all_losses, train_all_acc, trainloader, optimizer, criterion, split_train):
    model.train()
    # initial the running loss
    running_loss = 0.0
    # pick each data from trainloader i: batch index/ data: inputs and labels
    correct = 0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = torch.Tensor(labels)
        # print(type(labels))
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # print statistics
        running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update parameters
        optimizer.step()

        result = outputs > 0.5
        correct += (result == labels).sum().item()

        if i % 64 == 0:
            print('Training set: [Epoch: %d, Data: %6d] Loss: %.3f' %
                  (epochs + 1, i * 64, loss.item()))

    acc = correct / (split_train * 40)
    running_loss /= len(trainloader)
    train_all_losses.append(running_loss)
    train_all_acc.append(acc)
    print('\nTraining set: Epoch: %d, Accuracy: %.2f %%' % (epochs + 1, 100. * acc))


def validation(model, val_all_losses, val_all_acc, validloader, criterion):
    model.eval()
    validation_loss = 0.0
    correct = 0
    for data, target in validloader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = model(data)

        validation_loss += criterion(output, target).item()

        result = output > 0.5
        correct += (result == target).sum().item()

    validation_loss /= len(validloader)
    acc = correct / (len(validloader) * 40)

    val_all_losses.append(validation_loss)
    val_all_acc.append(acc)

    print('\nValidation set: Average loss: {:.3f}, Accuracy: {:.2f}%)\n'
          .format(validation_loss, 100. * acc))

    return acc


def test(model, attr_acc, attributes, testloader, criterion):
    test_loss = 0
    correct = 0
    pred = []
    for data, target in testloader:
        data = data.to('cuda')
        target = target.to('cuda')
        output = model(data)
        test_loss += criterion(output, target).item()

        result = output > 0.5
        correct += (result == target).sum().item()
        compare = (result == target)
        pred.append(compare[0])

    test_loss /= len(testloader)
    acc = correct / (len(testloader) * 40)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * acc))

    for m in range(len(attributes)):
        num = 0
        for n in range(len(pred)):
            if pred[n][m]:
                num += 1
        accuracy = num / len(pred)
        attr_acc.append(accuracy)

    for i in range(len(attr_acc)):
        print('Attribute: %s, Accuracy: %.3f' % (attributes[i], attr_acc[i]))


def main():
    # specifying the zip file name
    file_name = "./celeba/img_align_celeba.zip"

    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        if os.path.isdir('img_align_celeba') == 0:
            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall()
            print('Done!')
        else:
            print('File has already extracted.')

    data_path = sorted(glob.glob('img_align_celeba/*.jpg'))
    print(len(data_path))

    # get the label of images
    label_path = "./celeba/list_attr_celeba.txt"
    label_list = open(label_path).readlines()[2:]
    data_label = []
    for i in range(len(label_list)):
        data_label.append(label_list[i].split())

    # transform label into 0 and 1
    for m in range(len(data_label)):
        data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
        data_label[m] = [int(p) for p in data_label[m]]

    # get the attributes names for display
    attributes = open(label_path).readlines()[1].split()

    dataset = celeba(data_path, data_label)
    # split data into train, valid, test set 7:2:1
    indices = list(range(202599))
    split_train = 141819
    split_valid = 182339
    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=train_sampler)

    validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler)

    testloader = torch.utils.data.DataLoader(dataset, sampler=test_sampler)

    print(len(trainloader))
    print(len(validloader))
    print(len(testloader))

    # define empty list to store the losses and accuracy for ploting
    train_all_losses2 = []
    train_all_acc2 = []
    val_all_losses2 = []
    val_all_acc2 = []
    test_all_losses2 = 0.0

    # define the training epoches
    epochs = 20

    # instantiate Net class
    mobilenet = MobileNet()
    # use cuda to train the network
    mobilenet.to('cuda')
    # loss function and optimizer
    criterion = nn.BCELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(mobilenet.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    best_acc = 0.0

    for epoch in range(epochs):
        train(mobilenet, epoch, train_all_losses2, train_all_acc2, trainloader, optimizer, criterion, split_train)
        acc = validation(mobilenet, val_all_losses2, val_all_acc2, validloader, criterion)
        # record the best model
        if acc > best_acc:
            checkpoint_path = './model_checkpoint.pth'
            best_acc = acc
            # save the model and optimizer
            torch.save({'model_state_dict': mobilenet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            print('new best model saved')
        print("========================================================================")

    checkpoint_path = './model_checkpoint.pth'
    model = MobileNet().to('cuda')
    checkpoint = torch.load(checkpoint_path)
    print("model load successfully.")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    attr_acc = []
    test(model, attr_acc, attributes, testloader, criterion)

    # plot results
    plt.figure(figsize=(8, 10))
    plt.barh(range(40), [100 * acc for acc in attr_acc], tick_label=attributes, fc='brown')
    plt.savefig('./acc_barh.png')

    plt.figure(figsize=(8, 6))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True, linestyle='-.')
    plt.plot(train_all_losses2, c='salmon', label='Training Loss')
    plt.plot(val_all_losses2, c='brown', label='Validation Loss')
    plt.legend(fontsize='12', loc='upper right')
    plt.savefig('loss.png')

    plt.figure(figsize=(8, 6))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.grid(True, linestyle='-.')
    plt.plot(train_all_acc2, c='salmon', label='Training Accuracy')
    plt.plot(val_all_acc2, c='brown', label='Validation Accuracy')
    plt.legend(fontsize='12', loc='lower right')
    plt.savefig('./acc.png')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
