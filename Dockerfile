# Pull base image.
FROM ubuntu:18.04

LABEL maintainer="Raul Ivan Perez Martell <ivanpmartell@uvic.ca>"

# Install updates to base image
RUN \
  apt-get update -y \
  && apt-get install -y

RUN \
  apt-get install python3 python3-pip -y \
  && pip3 install virtualenv

RUN \
  adduser promoter

USER promoter

VOLUME ["/home/promoter/suprref"]

WORKDIR /home/promoter

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["suprref/entrypoint.sh"]