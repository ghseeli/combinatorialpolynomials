FROM sagemath/sagemath:latest

USER root
RUN apt-get update && \
    apt-get install -y normaliz && \
    sage -pip install pynormaliz

USER sage