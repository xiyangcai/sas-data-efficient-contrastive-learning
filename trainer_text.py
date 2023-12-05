from typing import List, Optional

from torch import Tensor, nn
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate.lbfgs_text import encode_train_set, train_clf, test_clf
from projection_heads.critic import LinearCritic
from trainer import Trainer


class NLPTrainer(Trainer):
    def __init__(self, num_positive=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_positive = num_positive

    #########################################
    #           Loss Functions              #
    #########################################
    def un_supcon_loss(self, z: Tensor, num_positive: int):
        batch_size = int(len(z) / num_positive)

        if self.distributed:
            all_z = [torch.zeros_like(z) for _ in range(self.world_size)]
            dist.all_gather(all_z, z)
            # Move all tensors to the same device
            aug_z = []
            for i in range(num_positive):
                aug_z.append([])
                for rank in range(self.world_size):
                    if rank == self.rank:
                        aug_z[-1].append(z[i * batch_size: (i + 1) * batch_size])
                    else:
                        aug_z[-1].append(all_z[rank][i * batch_size: (i + 1) * batch_size])
            z = [torch.cat(aug_z_i, dim=0) for aug_z_i in aug_z]
        else:
            aug_z = []
            idxs = torch.arange(0, len(z), num_positive)
            for i in range(num_positive):
                aug_z.append(z[idxs + i])
            z = aug_z

        sim = self.critic(z)
        # print(sim)
        log_sum_exp_sim = torch.log(torch.sum(torch.exp(sim), dim=1))
        # Positive Pairs Mask
        p_targets = torch.cat([torch.tensor(range(int(len(sim) / num_positive)))] * num_positive)
        # len(p_targets)
        pos_pairs = (p_targets.unsqueeze(1) == p_targets.unsqueeze(0)).to(self.device)
        # print(pos_pairs)
        inf_mask = (sim != float('-inf')).to(self.device)
        pos_pairs = torch.logical_and(pos_pairs, inf_mask)
        pos_count = torch.sum(pos_pairs, dim=1)
        pos_sims = torch.nansum(sim * pos_pairs, dim=-1)
        return torch.mean(-pos_sims / pos_count + log_sum_exp_sim)

    #########################################
    #           Train & Test Modules        #
    #########################################
    def train(self):
        self.net.train()
        self.critic.train()

        # Training Loop (over batches in epoch)
        train_loss = 0
        t = tqdm(enumerate(self.trainloader), desc='Loss', total=len(self.trainloader),
                 bar_format='{desc}{bar}{r_bar}')
        for batch_idx, inputs in t:
            num_positive = self.num_positive
            # x = torch.cat(inputs, dim=0).to(self.device)
            text, text_lengths = inputs.text

            self.encoder_optimizer.zero_grad()
            z = self.net(text, text_lengths)
            loss = self.un_supcon_loss(z, num_positive)
            loss.backward()

            self.encoder_optimizer.step()
            train_loss += loss.item()
            t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            print("lr:", self.scale_lr * self.lr_scheduler.get_last_lr()[0])

        return train_loss / len(self.trainloader)

    def test(self):
        X, y = encode_train_set(self.clftrainloader, self.device, self.net)
        representation_dim = self.net.module.representation_dim if self.distributed else self.net.representation_dim
        clf = train_clf(X, y, representation_dim, self.num_classes, self.device, reg_weight=1e-5, iter=100)
        acc = test_clf(self.testloader, self.device, self.net, clf)

        if acc > self.best_acc:
            self.best_acc = acc

        return acc

    def save_checkpoint(self, prefix):
        if self.world_size > 1:
            torch.save(self.net.module, f"{prefix}-net.pt")
        else:
            torch.save(self.net, f"{prefix}-net.pt")
        torch.save(self.critic, f"{prefix}-critic.pt")