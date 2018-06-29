FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update -y && apt-get upgrade -y

# Install python3 and pip3
RUN apt-get install -y python3 python3-pip python3-tk
RUN pip3 install --upgrade pip

# Move files to container
ADD . /forecast-engine
WORKDIR /forecast-engine

# Install required python dependencies
RUN pip3 install -r requirements.txt

EXPOSE 8050

CMD python3 app.py