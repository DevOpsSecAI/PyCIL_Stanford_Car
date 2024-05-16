from flask import Flask
import subprocess
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

logging.basicConfig(level=logging.DEBUG)

sentry_sdk.init(
    dsn="https://0dbd8387b1c5f375af0fe420ea5f15f2@o4507264673710080.ingest.us.sentry.io/4507264677969920",
    integrations=[
        LoggingIntegration(
            level=logging.DEBUG,  # Capture info and above as breadcrumbs
            event_level=logging.DEBUG,  # Send records as events
        ),
    ],
)

app = Flask(__name__)


@app.get("/")
def index():
    return "Hello, World!"

@app.get("/train")
def train():
    subprocess.call(["./entrypoint.sh"])
    return "Bash script triggered successfully!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
