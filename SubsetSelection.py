import torch
import random
from data_proc.dataset import IMDbDataset
from approx_latent_classes_text import glove_approx
from SubsetTextDataset import SASSubsetTextDataset

CRITIC_PATH = "2023-12-0317:57:33.875617-imdb-LSTM-99-critic.pt"
NET_PATH = "2023-12-0317:57:33.875617-imdb-LSTM-99-net.pt"
NUM_CLASSES = 2
SUBSET_FRAC = 0.2
def main():
    device = torch.device('cuda')
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

    from torch import nn

    class ProxyModel(nn.Module):
        def __init__(self, net, critic):
            super().__init__()
            self.net = net
            self.critic = critic

        def forward(self, text, text_lengths):
            return self.critic.project(self.net(text, text_lengths))

    net = torch.load(NET_PATH)
    critic = torch.load(CRITIC_PATH)
    proxy_model = ProxyModel(net, critic)

    subset_dataset = SASSubsetTextDataset(
        dataset=imdb_train_dataset,
        subset_fraction=SUBSET_FRAC,
        num_downstream_classes=NUM_CLASSES,
        device=device,
        proxy_model=proxy_model,
        approx_latent_class_partition=partition,
        verbose=True
    )

    subset_dataset.save_to_file("IMDb-0.2-sas-indices.pkl")


if __name__ == '__main__':
    main()
