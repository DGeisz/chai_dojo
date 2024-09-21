import yaml
import boto3

with open("~/chai_dojo/notebookes/creds.yaml", "r") as file:
    creds = yaml.safe_load(file)


s3 = boto3.client(
    "s3",
    aws_access_key_id=creds["access_key"],
    aws_secret_access_key=creds["secret_key"],
    region_name=creds["region"],
)
