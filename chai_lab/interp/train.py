import torch
import numpy as np

from time import time
from torch import optim

from chai_lab.interp.config import OSAEConfig, SAEConfig
from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.k_sae import KSae
from chai_lab.interp.o_sae import OSae


class SAETrainer:
    def __init__(self, cfg: SAEConfig, s3, suppress_logs=False):
        self.cfg = cfg
        self.s3 = s3

        print("Subtracting mean!")
        self.data_loader = DataLoader(
            batch_size=cfg.batch_size, s3=s3, subtract_mean=True
        )

        self.sae = KSae(cfg)
        self.loss_per_batch = []
        self.num_dead_per_batch = []

    def train(self, num_batches):
        print(f"Learning rate: {self.cfg.lr:0.2e}")
        optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

        dead_mask = torch.zeros(
            self.cfg.num_latents, dtype=torch.bool, device=self.cfg.device
        )

        total_feature_counts = torch.zeros(
            self.cfg.num_latents, dtype=torch.int32, device=self.cfg.device
        )

        start = time()

        for i in range(num_batches):
            batch = self.data_loader.next_batch()
            batch = batch.to(self.cfg.device)

            # (
            #     _sae_out,
            #     _pre_acts,
            #     _latent_acts,
            #     _latent_indices,
            #     fvu,
            #     feature_counts,
            #     auxk_loss,
            # ) = self.sae(batch, dead_mask)

            sae_output = self.sae(batch, dead_mask)

            total_feature_counts += sae_output.feature_counts

            if i > 0 and i % self.cfg.num_batches_for_dead_neuron_sample == 0:
                dead_mask = total_feature_counts == 0

                num_dead = dead_mask.sum().item()

                total_feature_counts = torch.zeros(
                    self.cfg.num_latents, dtype=torch.int32, device=self.cfg.device
                )

                self.num_dead_per_batch.append((i, num_dead))

            aux_loss = self.cfg.aux_fraction * sae_output.auxk_loss
            loss = sae_output.fvu + self.cfg.aux_fraction * sae_output.auxk_loss

            loss.backward()
            self.loss_per_batch.append(loss.item())

            self.sae.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            self.sae.set_decoder_norm_to_unit_norm()
            optimizer.zero_grad()

            if i % 50 == 0:
                num_dead = dead_mask.sum().item()

                print(
                    f"BATCH {i} {time() - start:0.3g} LOSS: {loss.item():.4g} | NUM DEAD: {num_dead} | NUM ALIVE: {self.cfg.num_latents - num_dead} | AUX: {aux_loss:.4g} | REGULAR: {sae_output.fvu.item():.4g}"
                )

                # print(
                #     f"BATCH {i} LOSS: {loss.item():.4g} | NUM DEAD: {num_dead} | NUM ALIVE: {self.cfg.num_latents - num_dead}"
                # )


class OSAETrainer:
    def __init__(self, cfg: OSAEConfig, s3):
        self.cfg = cfg
        self.s3 = s3

        self.data_loader = DataLoader(
            batch_size=cfg.batch_size,
            subtract_mean=cfg.subtract_mean,
            s3=s3,
        )

        self.osae = OSae(cfg, dtype=torch.bfloat16, data_loader=self.data_loader)
        self.loss_per_batch = []
        self._num_dead_per_batch = []

        self.aux_per_batch = []

    @property
    def num_dead_per_batch(self):
        return np.array(self._num_dead_per_batch).T

    def train(self, num_batches):
        print(f"Learning rate: {self.cfg.lr:0.2e}")
        optimizer = torch.optim.Adam(
            self.osae.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

        # Define number of batches to run before decay starts
        num_batches_before_decay = 100
        decay_rate = 0.99  # Rate of decay per batch
        final_lr = 0.001  # Final learning rate

        # Define a lambda function for the learning rate schedule
        def lr_lambda(batch_idx):
            if batch_idx < num_batches_before_decay:
                return 1.0  # Keep initial learning rate
            else:
                return max(
                    final_lr / optimizer.param_groups[0]["lr"],
                    decay_rate ** (batch_idx - num_batches_before_decay),
                )

        # Define the scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.dead_mask = self.osae.get_dead_feature_mask(
            torch.zeros(self.cfg.num_latents, dtype=torch.bool, device=self.cfg.device)
        )

        self.total_feature_counts = torch.zeros(
            self.cfg.num_latents, dtype=torch.int32, device=self.cfg.device
        )

        start = time()

        for i in range(num_batches):
            batch = self.data_loader.next_batch()
            batch = batch.to(self.cfg.device)

            sae_output = self.osae(batch, self.dead_mask)

            self.total_feature_counts += sae_output.feature_counts

            if i > 0 and i % self.cfg.num_batches_for_dead_neuron_sample == 0:
                self.dead_mask = self.osae.get_dead_feature_mask(
                    self.total_feature_counts == 0
                )

                num_dead = self.dead_mask.sum().item()

                self.total_feature_counts = torch.zeros(
                    self.cfg.num_latents, dtype=torch.int32, device=self.cfg.device
                )

                self._num_dead_per_batch.append((i, num_dead))

            aux_fraction = (
                0.0 if self.cfg.aux_fraction is None else self.cfg.aux_fraction
            )

            aux_loss = aux_fraction * sae_output.aux_fvu

            loss = sae_output.fvu + aux_loss

            loss.backward()
            self.loss_per_batch.append(loss.item())
            self.aux_per_batch.append(sae_output.aux_fvu.item())

            # self.osae.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            self.osae.set_decoder_norm_to_unit_norm()
            optimizer.zero_grad()

            if i % 50 == 0:
                num_dead = self.dead_mask.sum().item()

                print(
                    f"BATCH {i} {time() - start:0.3g} LOSS: {loss.item():.4g} | NUM DEAD: {num_dead} | NUM ALIVE: {self.cfg.num_latents - num_dead} | AUX: {aux_loss:.4g} | REGULAR: {sae_output.fvu.item():.4g}"
                )
