FROM pytorch/pytorch:latest
WORKDIR /app
COPY . /app

RUN pip install -r Asset/requirements.txt &&\
    pip install opencv-contrib-python ultralytics omegaconf lapx &&\
    apt-get update && apt-get install ffmpeg libsm6 libxext6 git  -y

CMD ["python", "app.py"]