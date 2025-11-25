import os
import boto3

endpoint = 'http://api.minio.encall.space'
print("Endpoint:", endpoint)

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id='NuA5nPuQfAVT3NWjHt4q',
    aws_secret_access_key='m4LemuRfS92ormtx1qp7LP0ThsjnsF5Hhuo6v4Rh',
)

print("Listing buckets:")
print(s3.list_buckets())

print("\nTry listing that artifacts prefix:")
resp = s3.list_objects_v2(
    Bucket="stockflow",
    Prefix="mlflow/2/models/m-b7140f53275f4d068314b98cefbedbb5/artifacts",
)
print("Result:", resp)