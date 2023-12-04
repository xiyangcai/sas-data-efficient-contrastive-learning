from torchtext import data
import torch


class IMDbDataLoader:
    def __init__(self, dataset, batch_size=64, pin_memory=True):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create BucketIterator with the correct sort_key
        self.iterator = data.BucketIterator(
            dataset=dataset,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.text),  # Add sort_key here
            sort_within_batch=True,
            device=self.device
        )

    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.iterator)
