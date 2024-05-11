FROM python:3.8.5

COPY . .

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y unzip

RUN pip install --no-cache-dir --upgrade pip

RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install Cython
RUN pip install -r requirements.txt

RUN chmod +x download_dataset.sh train.sh

RUN ./download_dataset.sh

ENTRYPOINT [ "./entrypoint.sh" ]