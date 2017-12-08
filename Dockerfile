FROM python:latest
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python", "segmentv2-api.py" ]