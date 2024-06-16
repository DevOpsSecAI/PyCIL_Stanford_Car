from flask import Flask, send_from_directory
from flask_autoindex import AutoIndex

import subprocess, os

app = Flask(__name__)
AutoIndex(app, browse_root=os.path.curdir)


@app.route("/train", methods=["GET"])
def train():
    try:
        subprocess.call(["./simple_train.sh"])
        return "Bash script triggered successfully!"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
