#FROM nvidia/cuda:8.0-devel-ubuntu16.04
FROM ubuntu:18.10
# conda deps
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    build-essential \
    gcc \
    gfortran \
    python3.6 \
    python3.6-dev \
    python3-pip \
    openssh-server &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
#RUN echo "--------============ python"
#RUN python3 --version
RUN pip3 install numpy
ARG ssh_prv_key
ARG ssh_pub_key
# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    libboost-serialization-dev  \
    libboost-iostreams-dev \
    openmpi-bin libopenmpi-dev

#RUN which cmake && \
#    cmake --version
#
#RUN which gcc && \
#    gcc --version
#
#RUN which mpicc && \
#    mpicc --version

RUN git clone -b feat/docker_ready --recursive git@github.com:Analytics4MD/A4MD-project-a4md.git a4md

#RUN echo "ll python include dir--------------=========================="
#RUN ls /usr/local/lib/python3.6/dist-packages/numpy/core/include

# Remove SSH keys
RUN rm -rf /root/.ssh/
#RUN echo "---===== CLoging done"
#RUN python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
#RUN python3 -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"
RUN cd a4md && \
    git checkout feat/new_metric_names && \
    mkdir build && \
    cd build && \
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=../_install &&\
    make install
