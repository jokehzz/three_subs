#FROM anibali/pytorch:cuda-10.1
#FROM anibali/pytorch:1.5.0-cuda10.2
FROM anibali/pytorch:1.8.1-cuda11.1
#FROM python:3.6

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com
USER root
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

