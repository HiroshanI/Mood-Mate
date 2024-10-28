FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y default-libmysqlclient-dev libgl1 pkg-config

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install tensorflow_intel
RUN pip install tf_keras==2.17.0
RUN pip install torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
