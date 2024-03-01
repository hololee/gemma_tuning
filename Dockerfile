FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANGUAGE=ko_KR.UTF-8

RUN apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y --no-install-recommends \
    git \
    vim \
    curl \
    ffmpeg \
    gcc \
    python3-pip \
    ca-certificates \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libssl-dev \
    libleveldb-dev \
    libgflags-dev 

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# pip server.
RUN mkdir ~/.pip
RUN echo "[global]"  >> ~/.pip/pip.conf
RUN echo "index-url=http://ftp.daumkakao.com/pypi/simple"  >> ~/.pip/pip.conf
RUN echo "trusted-host=ftp.daumkakao.com"  >> ~/.pip/pip.conf

RUN pip install --upgrade pip
RUN pip install setuptools pip-tools 

# # streamlit setting.
# ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true

# # jupyer notebook setting.
# RUN jupyter notebook --generate-config
# RUN echo "c.NotebookApp.allow_root=True" >> /root/.jupyter/jupyter_notebook_config.py

# git setting.
RUN curl -o ~/git-prompt.sh https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh && \
    echo "source ~/git-prompt.sh" >> ~/.bashrc && \ 
    echo "GIT_PS1_SHOWDIRTYSTATE=1" >> ~/.bashrc && \
    echo "PS1='\\u@\\h \\w\$(__git_ps1 \" (%s)\") $ '" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]
