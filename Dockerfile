FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y build-essential
RUN conda install -y cython numpy
RUN pip install nibabel==3.2.1 nilearn==0.8.0 popeye==1.0.0
