import os.path as ops
import numpy as np
import torch
import cv2
import torchvision


class TUSIMPLE(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, resize=(512, 256), flag='train'):
        self.root = root
        self.transforms = transforms
        self.resize = resize
        self.flag = flag

        self.img_pathes = []

        self.train_file = ops.join(root, 'train.txt')
        self.val_file = ops.join(root, 'val.txt')
        self.test_file = ops.join(root, 'test.txt')

        if self.flag == 'train':
            file_open = self.train_file
        elif self.flag == 'valid':
            file_open = self.val_file
        else:
            file_open = self.test_file

        with open(file_open, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split()
                self.img_pathes.append(line)

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, idx):
        gt_image = cv2.imread(self.img_pathes[idx][0], cv2.IMREAD_UNCHANGED)
        gt_binary_image = cv2.imread(self.img_pathes[idx][1], cv2.IMREAD_UNCHANGED)
        gt_instance = cv2.imread(self.img_pathes[idx][2], cv2.IMREAD_UNCHANGED)

        gt_image = cv2.resize(gt_image, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
        gt_binary_image = cv2.resize(gt_binary_image, dsize=self.resize, interpolation=cv2.INTER_NEAREST)
        gt_instance = cv2.resize(gt_instance, dsize=self.resize, interpolation=cv2.INTER_NEAREST)

        gt_image = gt_image / 127.5 - 1.0
        gt_binary_image = np.array(gt_binary_image / 255.0, dtype=np.uint8)
        gt_binary_image = gt_binary_image[:, :, np.newaxis]
        gt_instance = gt_instance[:, :, np.newaxis]

        gt_binary_image = np.transpose(gt_binary_image, (2, 0, 1))
        gt_instance = np.transpose(gt_instance, (2, 0, 1))

        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))
        # trsf = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False)
        # gt_image = trsf(gt_image)

        gt_binary_image = torch.tensor(gt_binary_image, dtype=torch.long).view(self.resize[1], self.resize[0])
        #gt_binary_image = torch.tensor(gt_binary_image, dtype=torch.float)
        # gt_instance = torch.tensor(gt_instance, dtype=torch.float)
        gt_instance = torch.tensor(gt_instance, dtype=torch.long).view(self.resize[1], self.resize[0])

        return gt_image, gt_binary_image, gt_instance

    
class TUSIMPLE_AUG(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, resize=(512, 256), flag='train'):
        self.root = root
        self.transforms = transforms
        self.resize = resize
        self.flag = flag

        self.img_pathes = []

        self.train_file = ops.join(root, 'train.txt')
        self.val_file = ops.join(root, 'val.txt')
        self.test_file = ops.join(root, 'test.txt')

        if self.flag == 'train':
            file_open = self.train_file
        elif self.flag == 'valid':
            file_open = self.val_file
        else:
            file_open = self.test_file

        with open(file_open, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split()
                self.img_pathes.append(line)

    def __len__(self):
        return len(self.img_pathes) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            gt_image = cv2.imread(self.img_pathes[int(idx/2)][0], cv2.IMREAD_UNCHANGED)
            gt_binary_image = cv2.imread(self.img_pathes[int(idx/2)][1], cv2.IMREAD_UNCHANGED)
            gt_instance = cv2.imread(self.img_pathes[int(idx/2)][2], cv2.IMREAD_UNCHANGED)
        else:
            gt_image = cv2.imread(self.img_pathes[int((idx-1)/2)][0], cv2.IMREAD_UNCHANGED)
            gt_binary_image = cv2.imread(self.img_pathes[int((idx-1)/2)][1], cv2.IMREAD_UNCHANGED)
            gt_instance = cv2.imread(self.img_pathes[int((idx-1)/2)][2], cv2.IMREAD_UNCHANGED)

            gt_image = cv2.flip(gt_image, 1)
            gt_binary_image = cv2.flip(gt_binary_image, 1)
            gt_instance = cv2.flip(gt_instance, 1)

        gt_image = cv2.resize(gt_image, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
        gt_binary_image = cv2.resize(gt_binary_image, dsize=self.resize, interpolation=cv2.INTER_NEAREST)
        gt_instance = cv2.resize(gt_instance, dsize=self.resize, interpolation=cv2.INTER_NEAREST)

        gt_image = gt_image / 127.5 - 1.0
        gt_binary_image = np.array(gt_binary_image / 255.0, dtype=np.uint8)
        gt_binary_image = gt_binary_image[:, :, np.newaxis]
        gt_instance = gt_instance[:, :, np.newaxis]

        gt_binary_image = np.transpose(gt_binary_image, (2, 0, 1))
        gt_instance = np.transpose(gt_instance, (2, 0, 1))

        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))
        # trsf = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False)
        # gt_image = trsf(gt_image)

        gt_binary_image = torch.tensor(gt_binary_image, dtype=torch.long).view(self.resize[1], self.resize[0])
        # gt_binary_image = torch.tensor(gt_binary_image, dtype=torch.float)
        # gt_instance = torch.tensor(gt_instance, dtype=torch.float)
        gt_instance = torch.tensor(gt_instance, dtype=torch.long).view(self.resize[1], self.resize[0])

        return gt_image, gt_binary_image, gt_instance