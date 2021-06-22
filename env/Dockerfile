FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

SHELL ["/bin/bash", "-c"]

COPY settings/ /opt/settings/

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
    vim \
    curl \
    bzip2 \
    dos2unix \
    patch \
    openssh-server

RUN pip install cmake
RUN pip install Cython
RUN pip install --ignore-installed -r /opt/settings/requirements.txt
RUN echo "root:root" | chpasswd

RUN cp -r /opt/settings/sshd_config /etc/ssh/ && \
    service ssh start

RUN jupyter notebook --generate-config && \
    cp -r /opt/settings/jupyter_notebook_config.py ~/.jupyter/

CMD ["/bin/bash", "-c", "service ssh restart && jupyter notebook --allow-root"]
