from chai_lab.interp import s3
from chai_lab.interp.config import SAEConfig
from chai_lab.interp.train import SAETrainer

cfg = SAEConfig(
    d_in=256,
    num_latents=256 * 32,  # 8192
    k=32,
    device="cuda:0",
    batch_size=2048,
    buffer_size_in_proteins=4,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
)


trainer = SAETrainer(cfg, s3=s3)

trainer.train(200)
