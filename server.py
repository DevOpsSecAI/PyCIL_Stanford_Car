from flask import Flask
import subprocess

app = Flask(__name__)


@app.route("/train", methods=["GET"])
def train():
    try:
        subprocess.call(["mv", "./car_data/car_data/test", "./car_data/car_data/val"])
        subprocess.call(
            [
                "python",
                "main.py",
                "--config",
                "./exps/simplecil_general.json",
                "--data",
                "./car_data/car_data",
            ]
        )
        return "Bash script triggered successfully!"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
