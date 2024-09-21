# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import matplotlib.pyplot as plt

from chai_lab.interp.s3 import s3_client
from chai_lab.interp.config import SAEConfig
from chai_lab.interp.train import SAETrainer

# %%
cfg = SAEConfig(
    d_in=256,
    num_latents=256 * 256,  
    k=32,
    device="cuda:0",
    batch_size=4096,
    buffer_size_in_proteins=16,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    
)
trainer = SAETrainer(cfg, s3=s3_client, suppress_logs=True)

# %%
trainer.train(1000)

# %%
plt.plot(trainer.loss_per_batch)





# %%
def plot_acts(activations, lim=None):
    plt.bar(range(activations.size(0)), activations.float().cpu().numpy() )

    if lim is not None:
        plt.ylim(-lim, lim)

# %%
acts = [trainer.shuffle_loader.next_batch() for _ in range(10)]

acts = torch.cat(acts, dim=0)

# %%

m = acts.mean(dim=0)
s = acts.std(dim=0)

# %%

plot_acts(s)


# %%
s.min()

# %%
m.abs().mean()



    # plt.show()

# %%
bb = trainer.shuffle_loader.next_batch()

# %%

cc = trainer.sae.normalize_and_center(bb)
out = trainer.sae(bb)[0]

# %%


i = 1
lim = 0.2
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plot_acts(cc[i], lim)
plt.title("Real")

plt.subplot(1, 3, 2)
plot_acts(out[i].detach(), lim)
plt.title("Reconstructed")

plt.subplot(1, 3, 3)
plot_acts(cc[i] - out[i].detach(), lim)  
plt.title("Difference")

print(cc[i].norm().item(), out[i].norm().item(), (cc[i] - out[i].detach()).norm().item())

plt.show()

# %%
top_v, top_i = trainer.sae.encode(bb)

# %%
data = torch.bincount(top_i.flatten(), minlength=cfg.num_latents).cpu().numpy()
data = data[data > 50]

plt.hist(data, bins=40, cumulative=-1)
plt.show()

# %%
am = data.argmax()

# %%
plot_acts(trainer.sae.W_dec[am + 1].detach(), lim=0.2)
plt.show()

# %%
top_v[0].sort().values













# %%
plot_acts(m)

# %%
plot_acts(s)

# %%
plot_acts(s / m.abs())

# %%
plt.hist(s.float().cpu().numpy(), bins=40)
plt.show()


# %%
mult = [trainer.shuffle_loader.next_batch() for _ in range(40)] 

means = torch.stack([m.mean(dim=0) for m in mult])

# %%
plot_acts(means.std(dim=0))

# %%
plt.hist((means.std(dim=0) / means.mean(dim=0)).detach().float().cpu().numpy(), bins=40)
plt.show()


# %%
# %%


# means[0].shape













# # %%
# acts

# # %%
# import matplotlib.pyplot as plt

# a = trainer.shuffle_loader.next_batch()
# out = trainer.sae(a)[0]
# # %%
# a = a / a.norm(dim=-1, keepdim=True)

# # %%

# i = 40

# plt.bar(range(a.size(1)), a[i].float().cpu().numpy())
# plt.show()

# # %%
# plt.bar(range(a.size(1)), out[i].float().detach().cpu().numpy())


# # %%
# a[0].float().cpu().numpy()

# # %%
# a[0].argmin()

# # %%
# b = trainer.shuffle_loader.next_batch()

# # %%
# big_small = b[:, 83]

# big_small.mean().item(), big_small.std().item(), big_small.max().item(), big_small.min().item()


# # %%
# c = b.clone()
# c[:, 83] = 0

# # %%
# c.norm(dim=-1).shape

# # %%
# d = c.norm(dim=-1) / big_small



# # %%
# plt.hist(d.float().detach().cpu().numpy(), bins=20)

# # %%
# plt.scatter(c.norm(dim=-1).float().detach().cpu().numpy(), big_small.float().detach().cpu().numpy() * -1)

# # %%

# %%
