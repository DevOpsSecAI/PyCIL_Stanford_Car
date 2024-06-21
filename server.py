from flask import Flask, send_from_directory
from flask_autoindex import AutoIndex

import subprocess, os

from download_s3_path import download_s3_folder
from split import split_files
import os

app = Flask(__name__)
AutoIndex(app, browse_root=os.path.curdir)


@app.route("/train", methods=["GET"])
def train():
    try:
        subprocess.call(["./simple_train.sh"])
        return "Bash script triggered successfully!"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {str(e)}", 500


@app.route("/train/workings/<working_id>", methods=["GET"])
def train_with_working_id(working_id):
    path = f"working/{str(working_id)}"
    download_s3_folder(os.getenv("S3_BUCKET_NAME", "pycil.com"), path, path)

    data_path = path + "/data"
    config_path = path + "/config.json"
    output_path = f"output/{working_id}"

    split_files(data_path)
    subprocess.call(
        [f"./train_from_working.sh {config_path} {data_path} {path} {output_path}"]
    )
    return f"Training started with working id {working_id}!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
