import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class LGASSNet_Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(LGASSNet_Dataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Lettuce/JPEGImages"), name + ".jpg"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "Lettuce/SegmentationClass"), name + ".png"))

        jpg, png = self.simple_resize(jpg, png, self.input_shape)

        jpg = np.transpose(self.preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def simple_resize(self, image, label, input_shape):
        h, w = input_shape
        image = image.resize((w, h), Image.BICUBIC)
        label = label.resize((w, h), Image.NEAREST)

        return image, label

    def preprocess_input(self, image):
        image /= 255.0
        return image


def LGASSNet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels