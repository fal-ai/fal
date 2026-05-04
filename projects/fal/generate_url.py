import json
import uuid

import boto3
from botocore.client import Config

BUCKET = "vedat-test"
REGION = "us-west-1"
PREFIX = f"uploads/req-{uuid.uuid4().hex[:8]}/"

s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=f"https://s3.{REGION}.amazonaws.com",
    config=Config(signature_version="s3v4"),
)

post = s3.generate_presigned_post(
    Bucket=BUCKET,
    Key=PREFIX + "${filename}",
    Fields={"success_action_status": "201"},
    Conditions=[
        {"success_action_status": "201"},
        ["starts-with", "$key", PREFIX],
        ["starts-with", "$Content-Type", ""],
        ["content-length-range", 0, 100 * 1024 * 1024],
    ],
    ExpiresIn=3600,
)
print(json.dumps(post))
