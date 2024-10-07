import torch
from chai_lab.interp.sae.o_sae import OSae
from chai_lab.interp.storage.s3 import s3_client


esm_sae = OSae(dtype=torch.bfloat16)
esm_sae.load_model_from_aws(s3_client, f"osae_1EN3_to_4EN2_{32 * 2048}.pth")

trunk_sae = OSae(dtype=torch.bfloat16)
trunk_sae.load_model_from_aws(s3_client, "osae_v1_1EN3_2EN2_65536_12000.pth")
