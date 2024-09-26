import torch

from chai_lab.interp.config import SAEConfig
from chai_lab.interp.data_loader import DataLoader
from chai_lab.interp.k_sae import KSae


class SAETrainer:
    def __init__(self, cfg: SAEConfig, s3, suppress_logs=False):
        self.cfg = cfg
        self.s3 = s3

        self.data_loader = DataLoader(
            batch_size=cfg.batch_size,
            s3=s3,
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
                    f"BATCH {i} LOSS: {loss.item():.4g} | NUM DEAD: {num_dead} | NUM ALIVE: {self.cfg.num_latents - num_dead}"
                )
