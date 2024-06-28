import boto3
import os


def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
    # Tạo S3 client
    s3 = boto3.client("s3")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    # Tải tệp từ S3 và lưu vào đường dẫn local
    s3.download_file(bucket_name, s3_file_key, local_file_path)

    # Trả về đường dẫn tuyệt đối của tệp đã tải xuống
    return os.path.abspath(local_file_path)


# # Ví dụ sử dụng
# bucket_name = "your-bucket-name"
# s3_file_key = "path/to/your/s3/file.txt"
# local_file_path = "D:/test/a.txt"

# downloaded_file_path = download_file_from_s3(bucket_name, s3_file_key, local_file_path)
# print(f"Downloaded file path: {downloaded_file_path}")
