from flask import Flask
import subprocess

app = Flask(__name__)


@app.get("/train")
def train():
    subprocess.call(["./train.sh", "./exps/simplecil.json"])
    return "Bash script triggered successfully!"


if __name__ == "__main__":
    # run subprocess.call(["./train.sh", "./exps/simplecil.json"]) in background after starting the server 3s
    subprocess.call(["./train.sh", "./exps/simplecil.json"])
