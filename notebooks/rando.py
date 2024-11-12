from typing import List
import torch
import einops
import math

from torch import nn
from torch.utils.data import DataLoader


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer1(x)


def train(dataloader: DataLoader, model: SimpleModel):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for x, y in dataloader:
        model = SimpleModel()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


class AttentionBlock(nn.Module):
    def __init__(self, d_hidden: int, num_heads: int):
        super().__init__()

        assert d_hidden % num_heads == 0

        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.d_head = d_hidden // num_heads

        self.Q = nn.Linear(d_hidden, d_hidden)
        self.K = nn.Linear(d_hidden, d_hidden)
        self.V = nn.Linear(d_hidden, d_hidden)
        self.O = nn.Linear(d_hidden, d_hidden)

        self.layer_norm = nn.LayerNorm(d_hidden)

    def causal_mask(self, num_tokens: int, rearrange_to_train_dim=True):
        base_mask = (
            torch.triu(torch.ones(num_tokens, num_tokens) * -float("inf"), diagonal=1),
        )

        if rearrange_to_train_dim:
            return einops.rearrange(
                base_mask,
                "x y, 1 1 x y",
            )
        else:
            return base_mask

    def forward(self, x):
        x = self.layer_norm(x)

        q, k, v = self.Q(x), self.K(x), self.V(x)

        q = einops.rearrange(
            q,
            "N seq (num_heads d_head) -> N seq num_heads d_head",
            num_tokens=self.num_heads,
        )
        k = einops.rearrange(
            k,
            "N seq (num_heads d_head) -> N seq num_heads d_head",
            num_tokens=self.num_heads,
        )
        v = einops.rearrange(
            v,
            "N seq (num_heads d_head) -> N seq num_heads d_head",
            num_tokens=self.num_heads,
        )

        pre_pattern = einops.einsum(
            q,
            k,
            "N x_seq num_heads d_head, N y_seq num_heads d_head -> N num_heads x_seq y_seq",
        ) / math.sqrt(self.d_head)

        attn_pattern = (pre_pattern + self.causal_mask(x.size(1))).softmax(dim=-1)

        z = einops.einsum(
            attn_pattern,
            v,
            "N num_heads x_seq y_seq, N y_seq num_heads d_head -> N x_seq num_heads d_head",
        )

        z = einops.rearrange(z, "N seq num_heads d_head -> N seq (num_heads d_head)")

        return self.O(z)


# # Q1: Implement a function that performs mini-batch gradient descent
# # given a dataset, batch size, learning rate, and number of epochs
# def mini_batch_gradient_descent(X_data, y_data, batch_size, learning_rate, epochs):
#     N = X_data.size(0)

#     for batch_start in range(0, N, batch_size):
#         batch_end = min(N, batch_start + batch_size)

#         X = X_data[batch_start:batch_end]
#         y = y_data[batch_start:batch_end]


# %%
def one_hot_encode(sequences: List[str], vocab: List[str]):
    vocab_index = {char: i for i, char in enumerate(vocab)}

    seq_i = torch.tensor([[vocab_index[char] for char in seq] for seq in sequences])

    one_hot = torch.zeros((len(sequences), len(sequences[0]), len(vocab)))

    one_hot[
        torch.arange(len(sequences)).unsqueeze(1),
        torch.arange(len(sequences[0])).unsqueeze(0),
        seq_i,
    ] = 1

    return one_hot


# %%
import torch
from typing import List


# %%
a = torch.tensor([[1, 2], [3, 6]])

# %%
a[[0, 1], [0, 0]] = 0


# %%

# %%
torch.diff(a, dim=0)

# %% 0
a = torch.tensor([[[1, 0, 3], [0, 1, 0]]])


# %%
a.sort(stable=True, dim=-2)

# %%
b = a - 1

# %%
b & a


# %%
torch.unravel_index(a)
# %%
torch.unravel_index(torch.tensor([1234]), (10, 10, 10, 10))

# %%
torch.ravel(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))

# %%

a = torch.zeros((3, 3, 3))
ind = torch.tensor([[[0, 2]], [[1, 0]]])
values = torch.tensor([[[1, 2]], [[3, 4]]], dtype=a.dtype)
result = a.scatter(-1, index=ind, src=torch.tensor(1, dtype=a.dtype))

result

# %%
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self):
        return 1


my_dataset = MyDataset()

dataloader = DataLoader(my_dataset, batch_size=10, shuffle=True)
