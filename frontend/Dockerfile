FROM python:3.10.12

Expose 8501

WORKDIR /app

RUN apt-get -y update &&\
    apt-get -y upgrade &&\
    apt-get install -y build-essential software-properties-common python3-requests
## TODO rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip3 install -r requirements.txt

COPY . /app


ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.address=0.0.0.0"]
