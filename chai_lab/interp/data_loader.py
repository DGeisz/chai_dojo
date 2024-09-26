import os
import torch
import io
import random

from tqdm import tqdm

from chai_lab.interp.s3_utils import (
    NUM_SUPER_BATCHES,
    bucket_name,
    get_local_filename,
    super_batch_s3_key,
)


def load_s3_object_with_progress(bucket_name, key, s3_client, chunk_size=1024 * 1024):
    res = s3_client.get_object(Bucket=bucket_name, Key=key)
    total_size = int(res["ContentLength"])

    # Create a BytesIO buffer to hold the data
    buffer = io.BytesIO()

    # Read data in chunks and show progress
    body = res["Body"]
    for chunk in tqdm(
        iter(lambda: body.read(chunk_size), b""),
        total=total_size // chunk_size,
        unit="MB",
        desc="Downloading",
    ):
        buffer.write(chunk)

    buffer.seek(0)  # Go back to the start of the buffer after writing
    return torch.load(buffer, map_location=torch.device("cpu"))  # or GPU if needed


def super_batch_file_name(index: int):
    return f"super_batch_{index}.pt2"


class DataLoader:
    def __init__(
        self,
        batch_size,
        subtract_mean,
        s3,
        bucket_name=bucket_name,
        suppress_logs=False,
    ):
        self.s3 = s3
        self.bucket_name = bucket_name
        self.suppress_logs = suppress_logs

        self.batch_size = batch_size
        self.batch_index = 0

        self.super_batch_index = 0
        self._load_super_batch(self.super_batch_index)

        self.mean = self.super_batch[:50_000].mean(dim=0)

        self.subtract_mean = subtract_mean

    def _load_super_batch(self, super_batch_index: int, chunk_size=1024 * 1024):
        key = super_batch_s3_key(super_batch_index)

        print(f"Starting Super Batch: load: {super_batch_index}")

        local_cache_file = get_local_filename(super_batch_file_name(super_batch_index))

        # Check if local cache file exists
        if not os.path.exists(local_cache_file):
            res = self.s3.get_object(Bucket=bucket_name, Key=key)
            total_size = int(res["ContentLength"])

            # Create a BytesIO buffer to hold the data
            buffer = io.BytesIO()

            # Read data in chunks and show progress
            body = res["Body"]
            for chunk in tqdm(
                iter(lambda: body.read(chunk_size), b""),
                total=total_size // chunk_size,
                unit="MB",
                desc="Downloading",
            ):
                buffer.write(chunk)

            buffer.seek(0)  # Go back to the start of the buffer after writing
            self.super_batch = torch.load(buffer)  # or GPU if needed

            # Save this locally
            torch.save(self.super_batch, local_cache_file)
        else:
            self.super_batch = torch.load(local_cache_file)

        print(f"Finished super batch load: {super_batch_index}")

    def get_normalized_mean(self, num_batches):
        acts = torch.cat([self.next_batch() for _ in range(num_batches)], dim=0)
        acts /= acts.norm(dim=-1, keepdim=True)

        return acts.mean(dim=0)

    def get_mean(self, num_batches):
        acts = torch.cat([self.next_batch() for _ in range(num_batches)], dim=0)

        return acts.mean(dim=0)

    def _inc_super_batch_index(self):
        self.super_batch_index = (self.super_batch_index + 1) % NUM_SUPER_BATCHES

    def next_batch(self):
        if (self.batch_index + 1) * self.batch_size >= self.super_batch.size(0):
            self._inc_super_batch_index()
            self._load_super_batch(self.super_batch_index)

            self.batch_index = 0

        start = self.batch_index * self.batch_size
        end = (self.batch_index + 1) * self.batch_size

        acts = self.super_batch[start:end]

        self.batch_index += 1

        if self.subtract_mean:
            return acts - self.mean
        else:
            return acts
