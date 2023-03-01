import os
import cv2
import shutil
import pandas as pd
import numpy as np
from easymia.core.abstract_dataset import Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from easymia.libs import manager
from easymia.transforms import Compose
from . import preprocess

@manager.DATASETS.add_component
class Gla_dataset(Dataset):
    """
    getitem() output:

    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)

        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                dataset_root,
                num_seg,
                val_ratio=None,
                img_transforms=None,
                oct_transforms=None,
                fundus_image_size=None,
                oct_image_size=None,
                label_file='',
                num_classes=3,
                split='train'
                ):
                
        self.dataset_root = dataset_root
        self.fundus_image_size = fundus_image_size
        self.oct_image_size = oct_image_size
        self.img_transforms = Compose('clas', img_transforms) \
                            if isinstance(img_transforms, (list, tuple)) else img_transforms
        self.oct_transforms = Compose('clas', oct_transforms) \
                            if isinstance(oct_transforms, (list, tuple)) else oct_transforms
        self.split = split.lower()
        self.num_classes = num_classes
        self.num_seg = num_seg

        if val_ratio is not None:
            train_filelists, val_filelists = self.do_filelist(dataset_root, val_ratio)
            if self.split == 'train':
                filelists = train_filelists
            elif self.split == 'val':
                filelists = val_filelists
        if self.split =='test' or val_ratio is None:
            filelists = os.listdir(dataset_root)
        if self.split in ['train', 'val']:
            label = {row['data']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root) ]
        elif self.split == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]
        if self.split == "train":
            self.file_list = self.over_sampling(self.file_list)      

    def do_filelist(self, dataset_root, val_ratio):
        filelists = os.listdir(dataset_root)
        train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
        print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))
        return train_filelists, val_filelists

    def over_sampling(self, file_lists):
        file_list = []
        label_list = []
        for files, labels in file_lists:
            file_list.append([files])
            label_list.append(labels.tolist())
        ros = RandomOverSampler(random_state=0)
        file_list, label_list = ros.fit_resample(np.array(file_list), np.array(label_list))
        file_lists = list(zip(file_list, label_list))
        for i, file_label in enumerate(file_lists):
            file_lists[i] = [file_label[0].tolist()[0], file_label[1]]
        return file_lists


    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]
        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        try:
            oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                        key=lambda x: int(x.split("_")[0]))
        except:
            
            shutil.rmtree(os.path.join(self.dataset_root, real_index, real_index, '.ipynb_checkpoints'))
            oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
                                        key=lambda x: int(x.split("_")[0]))
                                        
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR -> RGB
        fundus_img = self.fundus_preprocess(fundus_img, self.fundus_image_size)
        oct_list = list()
        num_frame = len(oct_series_list)
        if self.split in ['train', 'val']:
            clip_offsets = self._get_train_clips(num_frame)
        if self.split == 'test':
            clip_offsets = self._get_test_clips(num_frame)
        for idx in list(clip_offsets):
            oct_img = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[idx]),
                                  )[:, :, ::-1]
            oct_img = self.oct_preprocess(oct_img, self.oct_image_size)
            if self.oct_transforms is not None:
                oct_img = self.oct_transforms(oct_img)
            oct_list.append(oct_img)
        oct_list = np.array(oct_list).transpose(3, 0, 1, 2) 
        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)

        fundus_img = fundus_img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        if self.split == 'test':
            return fundus_img, oct_list, real_index
        if self.split in ['train', 'val']:
            label = label.argmax()
            return fundus_img, oct_list, label

    def oct_preprocess(self, img, image_size):
        img = preprocess.center_crop(img, 512)
        if image_size is not None:
            img = preprocess.resize(img, image_size)
        return img

    def fundus_preprocess(self, img, image_size):
        if image_size is not None: 
            img = preprocess.resize(img, (image_size, image_size))
        return img

    def _get_train_clips(self, num_frames):
        avg_interval = num_frames // self.num_seg

        if avg_interval > 0:
            base_offsets = np.arange(self.num_seg) * avg_interval
            clip_offsets = base_offsets + np.random.randint(avg_interval,
                                                            size=self.num_seg)
        return clip_offsets

    def _get_test_clips(self, num_frames):
        avg_interval = num_frames  / float(self.num_seg)
        base_offsets = np.arange(self.num_seg) * avg_interval
        clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        return clip_offsets

    def __len__(self):
        return len(self.file_list)
        