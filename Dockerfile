FROM python:3.8.5

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y unzip 

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install Cython

COPY --chown=user . .
RUN pip install -r requirements.txt

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app


RUN chmod +x download_dataset.sh

RUN ./download_dataset.sh

RUN chmod +x entrypoint.sh train.sh

ENTRYPOINT [ "./entrypoint.sh" ]
