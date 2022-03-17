from PIL import Image
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import check_integrity
import re

class GLAUCOMA(Dataset):
    base_folder = 'original_data'
    filename_train = 'glaucoma_rgb_train.npz'
    filename_val = 'glaucoma_rgb_val.npz'

    def __init__(self, root, train=True, transform=None):
        super(GLAUCOMA, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.classes = ['non-glaucoma', 'glaucoma']


        path_train = os.path.join(self.root, self.filename_train)
        path_val = os.path.join(self.root, self.filename_val)
        if not (check_integrity(path_train) & check_integrity(path_val)):
            self.preprocess_and_save_data()

        self.data, self.labels = self.load_data(self.train)
        self.labels = self.labels.tolist()


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self):
        return len(self.data)


    def load_data(self, train):
        if train:
            path = os.path.join(self.root, self.filename_train)
        else:
            path = os.path.join(self.root, self.filename_val)

        f = np.load(path)
        return f['x'], f['y']


    def preprocess_and_save_data(self):
        x, y = self.read_x_and_y()

        x, y = shuffle(x, y, random_state=5)

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=0.85, random_state=5)

        print('Glaucoma samples shapes x_train: {}, x_test: {}'.format(x_train.shape, x_val.shape))

        path_to_save_train = os.path.join(self.root, self.filename_train)
        path_to_save_val = os.path.join(self.root, self.filename_val)
        np.savez_compressed(path_to_save_train, x=x_train, y=y_train)
        np.savez_compressed(path_to_save_val, x=x_val, y=y_val)


    def read_x_and_y(self):
        array_of_images = []
        array_of_labels = []

        path_to_x = os.path.join(self.root, self.base_folder)
        for _, file in enumerate(os.listdir(path_to_x)):
            fname = path_to_x + '/' + file

            f_is_image = fname.endswith(('.jpg', '.jpeg', '.JPG', '.png', '.bmp'))
            f_has_label = (re.search('_g_', fname) is not None) | (re.search('Im[0-9]', fname) is not None)

            if f_is_image & f_has_label:
                image = cv2.imread(fname)
                single_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                single_array = cv2.resize(single_array, (256, 256), interpolation=cv2.INTER_AREA)
                array_of_images.append(single_array)

                if re.search('_g_', fname) is not None:
                    array_of_labels.append(1)
                else:
                    array_of_labels.append(0)

        array_of_images = np.array(array_of_images)
        array_of_labels = np.array(array_of_labels).astype(np.int32)
        return array_of_images, array_of_labels