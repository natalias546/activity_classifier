import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET   = "wisdm-dse"
REGION   = "us-west-2"
BASE     = os.path.dirname(os.path.abspath(__file__))
DEST     = os.path.join(BASE, "data", "wisdm-dataset", "raw")
SUBJECTS = range(1600, 1651)

SOURCES = [
    ("watch", "accel", "data_{}_accel_watch.txt"),
    ("watch", "gyro",  "data_{}_gyro_watch.txt"),
    ("phone", "accel", "data_{}_accel_phone.txt"),
    ("phone", "gyro",  "data_{}_gyro_phone.txt"),
]

# uses credentials from environment variables or ~/.aws/credentials
s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))

total = 0
for device, sensor, pattern in SOURCES:
    local_dir = os.path.join(DEST, device, sensor)
    os.makedirs(local_dir, exist_ok=True)
    for sid in SUBJECTS:
        filename   = pattern.format(sid)
        s3_key     = "raw/{}/{}/{}".format(device, sensor, filename)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            continue
        try:
            s3.download_file(BUCKET, s3_key, local_path)
            total += 1
            print("Downloaded", s3_key)
        except Exception as e:
            print("Skipped", s3_key, "-", e)

print("Done.", total, "files downloaded to", DEST)
