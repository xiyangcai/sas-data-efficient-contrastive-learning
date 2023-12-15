import os
from typing import Any, Callable, Optional

import nltk
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from torchtext import data, datasets
from tqdm import tqdm
import random
import string
import re


class CIFAR10Augment(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, transform=Callable, n_augmentations: int = 2, train: bool = True,
                 download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            List of augmented views of element at index
        """
        img = self.data[index]
        pil_img = Image.fromarray(img)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs


class STL10Augment(torchvision.datasets.STL10):
    def __init__(
            self,
            root: str,
            split: str,
            transform: Callable,
            n_augmentations: int = 2,
            download: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            download=download)
        self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs


class CIFAR100Augment(CIFAR10Augment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class ImageFolderAugment(ImageFolder):
    def __init__(self, root: str, transform=Callable, n_augmentations: int = 2):
        super().__init__(
            root=root,
            transform=transform,
        )
        self.n_augmentations = n_augmentations

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        pil_img = self.loader(path)
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class ImageNetAugment(torch.utils.data.Dataset):
    def __init__(self, root, transform, n_augmentations=2):
        self.root = root
        self.transform = transform
        self.n_augmentations = n_augmentations
        df = pd.read_csv(os.path.join(root, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pil_img = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        imgs = []
        for _ in range(self.n_augmentations):
            imgs.append(self.transform(pil_img))
        return imgs


############################################################################
# TEXT DATASETS
############################################################################

def preprocess_text(words):
    stop_words = set(stopwords.words('english'))
    stop_words.add('br')
    text = ' '.join(words)
    words = nltk.word_tokenize(text)
    # stemmer = SnowballStemmer("english")
    words = [re.sub('\W+', '', word) for word in words]
    words = [word.lower().replace(' ', '') for word in words if word.lower() not in stop_words]
    # stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return words


class IMDbDatasetSubset(torch.utils.data.Dataset):
    def __init__(self, idxs, max_vocab_size=20000, embedding_dim=100):
        self.TEXT = data.Field(tokenize='spacy', include_lengths=True)
        self.LABEL = data.LabelField(dtype=torch.float)

        train_data, _ = datasets.IMDB.splits(self.TEXT, self.LABEL)
        self.data = train_data
        print('train data size:', len(self.data))

        new_data = []
        for i in idxs:
            new_data.append(self.data[i])
        self.data.examples = new_data

        self.data = self.preprocess_dataset(self.data)

        self.TEXT.build_vocab(self.data, max_size=max_vocab_size, vectors=f"glove.6B.{embedding_dim}d")
        self.LABEL.build_vocab(self.data)

        self.fields = {'text': self.TEXT, 'label': self.LABEL}

    def preprocess_dataset(self, dataset):
        new_data = []
        for example in tqdm(dataset.examples, desc="Preprocessing text"):
            text = preprocess_text(example.text)
            label = example.label
            new_example = data.Example.fromlist(
                [text, label], [('text', self.TEXT), ('label', self.LABEL)]
            )
            new_data.append(new_example)
        dataset.examples = new_data
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', max_vocab_size=20000, embedding_dim=100):

        self.TEXT = data.Field(tokenize='spacy', include_lengths=True)
        self.LABEL = data.LabelField(dtype=torch.float)

        train_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(train_data, max_size=max_vocab_size, vectors=f"glove.6B.{embedding_dim}d")
        self.LABEL.build_vocab(train_data)

        if split == 'train':
            train_data = self.preprocess_dataset(train_data)
            self.data = train_data
        elif split == 'test':
            test_data = self.preprocess_dataset(test_data)
            self.data = test_data
        else:
            raise ValueError("Invalid split. Use 'train', or 'test'.")

        self.fields = {'text': self.TEXT, 'label': self.LABEL}

    def preprocess_dataset(self, dataset):
        new_data = []
        for example in tqdm(dataset.examples, desc="Preprocessing text"):
            text = preprocess_text(example.text)
            label = example.label
            new_example = data.Example.fromlist(
                [text, label], [('text', self.TEXT), ('label', self.LABEL)]
            )
            new_data.append(new_example)
        dataset.examples = new_data
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example.text
        label = example.label

        return data.Example.fromlist([text, label], fields=[('text', self.TEXT), ('label', self.LABEL)])


class AugmentedIMDbDataset(IMDbDataset):
    def __init__(self,
                 split='train',
                 max_vocab_size=20000,
                 embedding_dim=100,
                 augment_function=None,
                 num_positive=2,
                 indices=None,
                 aug_probs=0.5
                 ):
        super().__init__(split, max_vocab_size, embedding_dim)
        self.num_positive = num_positive
        self.augment_function = augment_function

        augmented_examples = []
        for idx, example in enumerate(self.data.examples):
            if indices is not None and idx not in indices:
                continue
            for _ in range(self.num_positive):
                example_augmented_text = self.augment_function(example.text, aug_probs)
                new_example = data.Example.fromlist(
                    [example_augmented_text, example.label],
                    [('text', self.TEXT), ('label', self.LABEL)]
                )
                augmented_examples.append(new_example)

        # self.TEXT.build_vocab(self.data, max_size=max_vocab_size, vectors=f"glove.6B.{embedding_dim}d")
        # self.LABEL.build_vocab(self.data)

        self.data.examples = augmented_examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
