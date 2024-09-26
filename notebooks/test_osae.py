# %%
%load_ext autoreload
%autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np

from chai_lab.interp.config import OSAEConfig
from chai_lab.interp.train import OSAETrainer
from chai_lab.interp.s3 import s3_client

# %%
cfg = OSAEConfig(
    k=32,
    # num_latents=256 * 256,
    num_latents=32 * 2048,
    # num_latents=32 * 1024,
    device="cuda:0",
    d_model=256,
    num_batches_for_dead_neuron_sample=20,
    batch_size=4096,
    lr=1e-2,
    beta1=0.9,
    beta2=0.999,
    aux_fraction=1/64,
    subtract_mean=True
    # aux_fraction=None
)

trainer = OSAETrainer(cfg, s3=s3_client)

# %%
trainer.train(9000)

# %%




# %%

plt.plot(trainer.loss_per_batch[100:])
plt.show()

# %%

x, y = trainer.num_dead_per_batch
plt.plot(x, y)

# %%

# %%
# train

# %%
import einops
import torch

x = trainer.data_loader.next_batch().to(trainer.osae.device)
osae = trainer.osae



x -= osae.b_dec

acts = einops.einsum(
    x, osae.encoder, "batch d_model, k d_group d_model -> batch k d_group"
)

acts += osae.b_enc

out, _max_values, max_indices = osae.decode_from_acts(acts)

out += osae.b_dec

e = (x - out).pow(2).sum()
total_variance = (x - x.mean(0)).pow(2).sum()

fvu = e / total_variance

feature_counts = osae.get_feature_counts(max_indices)

# %%


dead_mask = trainer.dead_mask

total_feature_counts = trainer.total_feature_counts

# %%
dead_mask.any(dim=-1).all()

# %%



if osae.cfg.aux_fraction and dead_mask is not None and dead_mask.any():
    aux_acts = torch.where(dead_mask[None], acts, -torch.inf)
    aux_out, _, _ = osae.decode_from_acts(aux_acts)

    e_hat = (aux_out - x).pow(2).sum()
    aux_fvu = e_hat / total_variance
else:
    aux_fvu = torch.tensor(0.0, device=osae.device)

# %%
# Check if dead_mask has any True
dead_mask.any()
    


# %%
trainer.train(1000)


# %%

plt.plot(trainer.loss_per_batch[:100])
plt.show()

# %%
plt.plot(trainer.num_dead_per_batch[1])

# %%
trainer.num_dead_per_batch[]


