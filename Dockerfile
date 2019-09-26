FROM python:3.7

RUN pip3 install --upgrade pip

RUN mkdir -p /home/ml-pipeline
RUN chmod -R 777 /home/ml-pipeline

WORKDIR /

COPY . /home/ml-pipeline
RUN chmod -R 777 /home/ml-pipeline

WORKDIR /home/ml-pipeline

RUN pip3 install -r requirements.txt
RUN jupyter nbconvert --to html ml-pipeline-slides.ipynb

EXPOSE 8000

# ENTRYPOINT ["jupyter", "nbconvert", "ml-pipeline-slides.ipynb", "--to", "slides", "--post", "serve"]
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]