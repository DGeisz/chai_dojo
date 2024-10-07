import torch

from functools import partial
from dataclasses import dataclass

from chai_lab.interp.max_acts.max_acts_aggregation import (
    create_flat_coords,
    group_and_sort_activations,
    init_aggregators,
    update_aggregators,
)


def flat_index_full(x, y, k_i, N, k):
    return x * N * k + y * k + k_i


def test_create_flat_coords():
    N = 4
    k = 5
    fasta_index = 1

    flat_coords = create_flat_coords(N, k, fasta_index)

    flat_index = partial(flat_index_full, N=N, k=k)

    for x in range(N):
        for y in range(N):
            for k_i in range(k):
                coords = flat_coords[flat_index(x, y, k_i)]
                coords = tuple(coords.tolist())

                assert coords == (x, y, fasta_index)


@dataclass
class CoordValue:
    x: int
    y: int
    k_i: int
    fasta_index: int

    feature_index: int
    value: float

    def to_coord_tuple(self):
        return (self.x, self.y, self.fasta_index)


def test_group_sort_activations():
    N = 4
    k = 5
    fasta_index = 1

    flat_index = partial(flat_index_full, N=N, k=k)

    flat_act_values = torch.zeros(N * N * k).float()
    flat_act_indices = 40 * torch.ones(N * N * k).int()

    sort_buckets = [
        [
            CoordValue(
                x=1, y=3, k_i=3, fasta_index=fasta_index, feature_index=1, value=10.0
            ),
            CoordValue(
                x=2, y=0, k_i=0, fasta_index=fasta_index, feature_index=1, value=8.0
            ),
        ],
        [
            CoordValue(
                x=0, y=2, k_i=4, fasta_index=fasta_index, feature_index=3, value=20.0
            ),
            CoordValue(
                x=0, y=0, k_i=2, fasta_index=fasta_index, feature_index=3, value=12.0
            ),
        ],
    ]

    flat_buckets = [c for bucket in sort_buckets for c in bucket]

    for c in flat_buckets:
        fi = flat_index(*c.to_coord_tuple())

        flat_act_values[fi] = c.value
        flat_act_indices[fi] = c.feature_index

    flat_coords = create_flat_coords(N, k, fasta_index)

    unique_buckets, value_buckets, coord_buckets = group_and_sort_activations(
        flat_act_values, flat_act_indices, flat_coords
    )

    assert torch.equal(unique_buckets, torch.tensor([1, 3, 40]))

    for i, bucket in enumerate(sort_buckets):
        value_bucket = value_buckets[i]
        coord_bucket = coord_buckets[i]

        for j, c in enumerate(bucket):
            assert c.value == value_bucket[j].item()
            assert c.to_coord_tuple() == tuple(coord_bucket[j].tolist())


def test_update_aggregators():
    N = 4
    k = 5
    num_latents = 100
    n = 4

    value_aggregator = -1 * torch.ones((num_latents, 2 * n)).float()
    coord_aggregator = -1 * torch.ones((num_latents, 2 * n, 3)).int()

    sort_groups = [
        [
            # Feature index 1
            CoordValue(x=1, y=3, k_i=3, fasta_index=1, feature_index=1, value=20.0),
            CoordValue(x=0, y=3, k_i=3, fasta_index=1, feature_index=1, value=10.0),
            CoordValue(x=1, y=2, k_i=4, fasta_index=1, feature_index=1, value=5.0),
            CoordValue(x=1, y=0, k_i=0, fasta_index=1, feature_index=1, value=1.0),
            # Feature index 3
            CoordValue(x=0, y=2, k_i=4, fasta_index=1, feature_index=3, value=10.0),
        ],
        [
            CoordValue(x=1, y=3, k_i=3, fasta_index=2, feature_index=1, value=12.0),
            CoordValue(x=0, y=3, k_i=3, fasta_index=2, feature_index=1, value=3.0),
            CoordValue(x=1, y=2, k_i=4, fasta_index=2, feature_index=1, value=4.0),
            CoordValue(x=0, y=0, k_i=0, fasta_index=2, feature_index=1, value=1.0),
        ],
        [
            # Feature index 1
            CoordValue(x=1, y=3, k_i=3, fasta_index=3, feature_index=1, value=3.0),
            CoordValue(x=0, y=3, k_i=3, fasta_index=3, feature_index=1, value=18.0),
            CoordValue(x=1, y=2, k_i=4, fasta_index=3, feature_index=1, value=2.5),
            CoordValue(x=0, y=0, k_i=0, fasta_index=3, feature_index=1, value=1.0),
            # Feature index 3
            CoordValue(x=2, y=3, k_i=2, fasta_index=3, feature_index=3, value=8.0),
        ],
    ]

    value_aggregator, coord_aggregator = init_aggregators(num_latents, n)

    flat_index = partial(flat_index_full, N=N, k=k)

    for group in sort_groups:
        flat_act_values = torch.zeros(N * N * k).float()
        flat_act_indices = 40 * torch.ones(N * N * k).int()

        for c in group:
            fi = flat_index(c.x, c.y, c.k_i)

            flat_act_values[fi] = c.value
            flat_act_indices[fi] = c.feature_index

        flat_coords = create_flat_coords(N, k, c.fasta_index)

        unique_buckets, value_buckets, coord_buckets = group_and_sort_activations(
            flat_act_values, flat_act_indices, flat_coords
        )

        value_aggregator, coord_aggregator = update_aggregators(
            value_aggregator,
            coord_aggregator,
            unique_buckets,
            value_buckets,
            coord_buckets,
        )

    for feature_index, query_num in [(1, n), (3, 2)]:
        flat_coords = [
            c
            for bucket in sort_groups
            for c in bucket
            if c.feature_index == feature_index
        ]

        # Sort flat_coords by value
        flat_coords.sort(key=lambda c: c.value, reverse=True)

        sorted_coords = flat_coords[:query_num]

        for i, c in enumerate(sorted_coords):
            value_aggregator_i = value_aggregator[c.feature_index]
            coord_aggregator_i = coord_aggregator[c.feature_index]

            assert c.value == value_aggregator_i[i].item()
            assert c.to_coord_tuple() == tuple(coord_aggregator_i[i].tolist())
