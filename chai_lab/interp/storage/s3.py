import yaml
import boto3
import os

creds_path = os.path.expanduser("~/chai_dojo/notebooks/creds.yaml")

with open(creds_path, "r") as file:
    creds = yaml.safe_load(file)


def create_new_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        region_name=creds["region"],
    )


s3_client = create_new_s3_client()
