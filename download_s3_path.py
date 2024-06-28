import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def download_s3_folder(bucket_name, s3_folder, local_dir):
    # Convert local_dir to an absolute path
    local_dir = os.path.abspath(local_dir)

    # Ensure local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client('s3')

    try:
        # List objects within the specified folder
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
        if 'Contents' not in objects:
            print(f"The folder '{s3_folder}' does not contain any files.")
            return

        for obj in objects['Contents']:
            # Formulate the local file path
            s3_file_path = obj['Key']
            if s3_file_path.endswith('/'):
                # Skip directories
                continue

            local_file_path = os.path.join(local_dir, os.path.relpath(s3_file_path, s3_folder))

            # Create local directories if they do not exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3.download_file(bucket_name, s3_file_path, local_file_path)
            print(f'Downloaded {s3_file_path} to {local_file_path}')
    
    except KeyError:
        print(f"The folder '{s3_folder}' does not contain any files.")
    except NoCredentialsError:
        print("Credentials not available.")
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
    except PermissionError as e:
        print(f"Permission error: {e}. Please check your directory permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download an S3 folder to a local directory.')
    parser.add_argument('-bucket', type=str, required=True, help='The S3 bucket name.')
    parser.add_argument('-s3_folder', type=str, required=True, help='The folder path within the S3 bucket.')
    parser.add_argument('-local_dir', type=str, required=True, help='The local directory to download the files to.')
    args = parser.parse_args()

    download_s3_folder(args.bucket, args.s3_folder, args.local_dir)
