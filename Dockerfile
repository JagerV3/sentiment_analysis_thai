FROM python:latest
# FROM ubuntu:latest
# RUN apt-get update -y
# RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python", "segmentv2-api.py" ]