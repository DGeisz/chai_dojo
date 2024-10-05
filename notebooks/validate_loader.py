# %%
%load_ext autoreload
%autoreload 2

# %%
import yaml
import boto3

from chai_lab.interp.data.shuffle_loader import PairActivationShuffleLoader
# %%
with open("creds.yaml", "r") as file:
    creds = yaml.safe_load(file)

batch_size = 256


s3 = boto3.client(
    "s3",
    aws_access_key_id=creds["access_key"],
    aws_secret_access_key=creds["secret_key"],
    region_name=creds["region"],
)

# %%
s3.download_file('mech-interp', 'chai/acts/102l_acts.pt2', '102l_acts.pt2')

# %%
import torch
a = torch.load('102l_acts.pt2')

# %%
a





# %%
res = s3.get_object(Bucket="mech-interp", Key="chai/acts/102l_acts.pt2")

stuff = res["Body"].read()

# %%
shuffle_loader = PairActivationShuffleLoader(
    batch_size=batch_size,
    s3=s3,
)

# %%
b = shuffle_loader.next_batch()




# %%
b.std()
