import torch
import random
from data_proc.dataset import IMDbDataset
from approx_latent_classes_text import glove_approx
from SubsetTextDataset import SASSubsetTextDataset
from torch import nn
import argparse

CRITIC_PATH = "emb300-hidden512-full-RandCrop0.2-critic.pt"
NET_PATH = "emb300-hidden512-full-RandCrop0.2-net.pt"
NUM_CLASSES = 2
# SUBSET_FRAC = 0.8


class ProxyModel(nn.Module):
    def __init__(self, net, critic):
        super().__init__()
        self.net = net
        self.critic = critic

    def forward(self, text, text_lengths):
        return self.critic.project(self.net(text, text_lengths))


def main(args):
    device = torch.device('cuda')
    subset_frac = args.subset_fraction
    imdb_train_dataset = IMDbDataset(split='train')

    rand_labeled_examples_indices = random.sample(range(len(imdb_train_dataset)), 500)
    rand_labeled_examples_labels = [
        1 if imdb_train_dataset[i].label == 'pos' else 0 for i in rand_labeled_examples_indices
    ]

    partition = glove_approx(
        trainset=imdb_train_dataset,
        labeled_example_indices=rand_labeled_examples_indices,
        labeled_examples_labels=rand_labeled_examples_labels,
        num_classes=NUM_CLASSES,
        device=device
    )

    net = torch.load(NET_PATH)
    critic = torch.load(CRITIC_PATH)
    proxy_model = ProxyModel(net, critic)

    subset_dataset = SASSubsetTextDataset(
        dataset=imdb_train_dataset,
        subset_fraction=subset_frac,
        num_downstream_classes=NUM_CLASSES,
        device=device,
        proxy_model=proxy_model,
        approx_latent_class_partition=partition,
        verbose=True
    )

    subset_dataset.save_to_file(f"IMDb-{subset_frac}-sas-indices.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAS Selection')
    parser.add_argument('--subset-fraction', type=float,
                        help="Size of Subset as fraction")

    args = parser.parse_args()
    main(args)
