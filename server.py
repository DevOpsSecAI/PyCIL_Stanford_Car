from flask import Flask, send_from_directory, request, send_file
from flask_autoindex import AutoIndex
import subprocess, os

from download_s3_path import download_s3_folder
from download_file_from_s3 import download_from_s3
from split import split_data
import os
import shutil
import json

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "upload"
AutoIndex(app, browse_root=os.path.curdir)


@app.route("/train", methods=["GET"])
def train():
    try:
        subprocess.Popen(["./simple_train.sh"])
        return "Bash script triggered successfully!"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {str(e)}", 500


@app.route("/train/workings/<working_id>", methods=["GET"])
def train_with_working_id(working_id):
    path = f"working/{working_id}"
    delete_folder(path)
    download_s3_folder(os.getenv("S3_BUCKET_NAME", "pycil.com"), path, path)

    data_path = path + "/data"
    config_path = path + "/config.json"
    output_path = f"s3://pycil.com/output/{working_id}"

    f = open(config_path, "r")
    args = json.load(f)

    args["data"] = request.args.get("data")

    if args["data"] is None:
        return "Data is not provided", 400

    output_model_path = "models/{}/{}_{}/{}/{}".format(
        args["model_name"], args["dataset"], args["data"], init_cls, args["increment"]
    )
    split_data(data_path)

    subprocess.Popen(
        [
            "./train_from_working.sh",
            config_path,
            data_path,
            output_model_path,
            f"s3://pycil.com/{path}",
        ]
    )

    return f"Training started with working id {working_id}!"


@app.route("/inference", methods=["POST"])
def infernece():
    file = request.files["image"]
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    config_path = request.form["config_path"]
    checkpoint_path = request.form["checkpoint_path"]

    download_from_s3("pycil.com", config_path, "config.json")
    download_from_s3("pycil.com", checkpoint_path, "checkpoint.pkl")
    subprocess.call(
        [
            "python",
            "inference.py",
            "--config",
            "config.json",
            "--checkpoint",
            "checkpoint.pkl",
            "--input",
            input_path,
            "--output",
            "output.json",
        ]
    )
    return send_file("output.json")


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
