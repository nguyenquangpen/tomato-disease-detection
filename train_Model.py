"""
@author: Quang Nguyen <nguyenquangpen@gmail.com>
"""

import os
import torch
from torchvision.transforms import ColorJitter, RandomAffine, Compose, Resize, ToTensor
from torch.utils.data import DataLoader, dataset
from GetDataTomato import DataTomato
from MyModel import Model
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import shutil

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    row_sums = cm.sum(axis=1)
    if np.any(row_sums == 0):
        row_sums[row_sums == 0] = 1
    cm = np.around(cm.astype('float') / row_sums[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parer = ArgumentParser(description='CNN training')
    parer.add_argument('--root', '-r', type=str, help='root folder')
    parer.add_argument('--epoch', '-e', type=int, default=100, help='number of epoch')
    parer.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parer.add_argument('--image_size', '-i', type=int, default=224, help='image size')
    parer.add_argument('--logging', '-l', type=str, default='tensorBoard')
    parer.add_argument('--train_model', '-t', type=str, default='trained_model')
    parer.add_argument('--checkpoint', '-c', type=str, default=None)
    args = parer.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.root = '/mnt/d/DataDeepLearning/PlantVillage'

    train_transforms = Compose([
        RandomAffine(
            degrees=(-4, 3.5),
            translate=(0.05, 0.05),
            scale=(0.85, 1.15),
            shear = 6,
        ),
        Resize((args.image_size, args.image_size)),
        ColorJitter(
            brightness=0.20,
            contrast=0.1,
            saturation=0.125,
            hue = 0.05
        ),
        ToTensor()
    ])

    test_transforms = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    train_data = DataTomato(root=args.root, train=True, transform=train_transforms)
    # image, label = train_data.__getitem__(3000)
    # image.show()
    train_dataLoader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    test_data = DataTomato(root=args.root, train=False, transform=test_transforms)
    test_dataLoader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )
    if not os.path.exists(args.train_model):
        os.makedirs(args.train_model)

    model = Model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0)
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    writer = SummaryWriter(args.logging)

    for epoch in range(start_epoch, args.epoch):
        model.train()
        progress_bar = tqdm(train_dataLoader, colour='green')
        for iter, (image, label) in enumerate(progress_bar):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)

            progress_bar.set_description(
                'epoch {}/{}. Interation {}/{}. loss {:.3f}'.format(epoch + 1, args.epoch, iter + 1,
                                                                    len(train_dataLoader), loss))
            writer.add_scalar('train/loss', loss, epoch * len(train_dataLoader) + iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        all_predic = []
        all_label = []

        for image, label in test_dataLoader:
            all_label.extend(label)
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = model(image)
                all_predic.extend(output.argmax(dim=1))
        all_label = [label.item() for label in all_label]
        all_predic = [predic.item() for predic in all_predic]
        plot_confusion_matrix(writer, confusion_matrix(all_label, all_predic), test_data.catergories, epoch)
        accu = accuracy_score(all_label, all_predic)
        print(accu)
        writer.add_scalar('test/accuracy', accu, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, '{}/Last_cnn.pth'.format(args.train_model))

        if accu > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": accu
            }
            torch.save(checkpoint, '{}/Best_cnn.pth'.format(args.train_model))
            best_acc = accu
