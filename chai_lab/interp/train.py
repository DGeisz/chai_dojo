import torch

from chai_lab.interp.config import SAEConfig
from chai_lab.interp.k_sae import KSae
from chai_lab.interp.shuffle_loader import PairActivationShuffleLoader


class SAETrainer:
    def __init__(self, cfg: SAEConfig, s3, suppress_logs=False):
        self.cfg = cfg
        self.s3 = s3

        self.shuffle_loader = PairActivationShuffleLoader(
            batch_size=cfg.batch_size,
            s3=s3,
            buffer_size_in_proteins=cfg.buffer_size_in_proteins,
            suppress_logs=suppress_logs,
        )

        self.sae = KSae(cfg, self.shuffle_loader.get_mean(40))
        self.loss_per_batch = []

    def train(self, num_batches):
        print(f"Learning rate: {self.cfg.lr:0.2e}")
        optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

        for i in range(num_batches):
            batch = self.shuffle_loader.next_batch()
            batch = batch.to(self.cfg.device)

            _, _, _, loss = self.sae(batch)
            loss.backward()
            self.loss_per_batch.append(loss.item())

            self.sae.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            self.sae.set_decoder_norm_to_unit_norm()
            optimizer.zero_grad()

            if i % 50 == 0:
                print(f"Batch {i} loss: {loss.item()}")
            if i % 100 == 0:
                print(
                    f"Num dead: {self.sae.get_num_dead(10, self.shuffle_loader.next_batch)}"
                )
