FROM python:3.8.5

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME

RUN apt-get update && apt-get install -y unzip 

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY --chown=user requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY --chown=user download_dataset.sh download_dataset.sh

RUN chmod +x download_dataset.sh

RUN ./download_dataset.sh

COPY --chown=user . .

RUN chmod +x entrypoint.sh train.sh install_awscli.sh

RUN ./install_awscli.sh

CMD ["python", "server.py"]
