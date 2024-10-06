import torch
from chai_lab.interp.sae.o_sae import OSae
from chai_lab.interp.storage.s3 import s3_client


trained_sae = OSae(dtype=torch.bfloat16)
trained_sae.load_model_from_aws(s3_client, f"osae_1EN3_to_4EN2_{32 * 2048}.pth")
