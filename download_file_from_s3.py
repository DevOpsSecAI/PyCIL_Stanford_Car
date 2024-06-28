import os
import boto3
from botocore.exceptions import NoCredentialsError


def download_from_s3(bucket_name, s3_key, local_path, is_directory=False):
    """
    Download a file or directory from S3 to a local path.

    :param bucket_name: str. The name of the S3 bucket.
    :param s3_key: str. The S3 key (path to the file or directory).
    :param local_path: str. The local file path or directory to download to.
    :param is_directory: bool. Set to True if s3_key is a directory.
    """
    s3 = boto3.client("s3")

    if is_directory:
        # Ensure the local directory exists
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        # List all objects in the specified S3 directory
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_key)
        print(result)

        if "Contents" in result:
            for obj in result["Contents"]:
                s3_object_key = obj["Key"]
                # Remove the directory prefix to get the relative file path
                relative_path = os.path.relpath(s3_object_key, s3_key)
                local_file_path = os.path.join(local_path, relative_path)

                # Ensure the local directory for the file exists
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)

                # Download the file
                s3.download_file(bucket_name, s3_object_key, local_file_path)
                print(f"Downloaded {s3_object_key} to {local_file_path}")
    else:
        # Download a single file
        print(f"Downloaded {s3_key} to {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)


# Example usage:
# download_from_s3('my-bucket', 'path/to/myfile.txt', 'local/path/to/myfile.txt')
# download_from_s3('my-bucket', 'path/to/mydirectory/', 'local/path/to/mydirectory', is_directory=True)
