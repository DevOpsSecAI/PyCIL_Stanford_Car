from flask import Flask
import subprocess

app = Flask(__name__)


@app.get("/train")
def train():
    subprocess.call(["./train.sh", "./exps/simplecil.json"])
    return "Bash script triggered successfully!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)