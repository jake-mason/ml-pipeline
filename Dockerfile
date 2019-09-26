FROM python:3.7

RUN pip3 install --upgrade pip

RUN mkdir -p /home/ml-pipeline
RUN chmod -R 777 /home/ml-pipeline

WORKDIR /

COPY . /home/ml-pipeline
RUN chmod -R 777 /home/ml-pipeline

WORKDIR /home/ml-pipeline

RUN pip3 install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]