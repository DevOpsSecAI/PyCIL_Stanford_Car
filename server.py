from flask import Flask
import subprocess

app = Flask(__name__)

@app.get('/')
def index():
    subprocess.call(['./entrypoint.sh'])
    return 'Bash script triggered successfully!'

if __name__ == '__main__':
    app.run()