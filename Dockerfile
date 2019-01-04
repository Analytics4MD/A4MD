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
RUN echo "--------============ python"
RUN python3 --version
RUN pip3 install numpy
#ENV CONDA_DIR="/opt/conda"
#ENV PATH="$CONDA_DIR/bin:$PATH"
#
## conda 
#RUN CONDA_VERSION="4.5.11" && \
#    CONDA_MD5_CHECKSUM="e1045ee415162f944b6aebfe560b8fee" && \
#    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O miniconda.sh && \
#    echo "$CONDA_MD5_CHECKSUM  miniconda.sh" | md5sum -c && \
#    /bin/bash miniconda.sh -f -b -p "$CONDA_DIR" && \
#    rm miniconda.sh && \
#    conda install python=3.5 && \
#    conda update --all --yes && \
#    conda config --set auto_update_conda False && \
#    /opt/conda/bin/conda clean -tipsy && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc && \
#    hash -r

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



# make sure your domain is accepted
#RUN touch /root/.ssh/known_hosts
#RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

#RUN pip3 install --upgrade cmake
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    libboost-serialization-dev  \
    libboost-iostreams-dev \
    openmpi-bin libopenmpi-dev

RUN which cmake && \
    cmake --version

RUN which gcc && \
    gcc --version

RUN which mpicc && \
    mpicc --version

RUN git clone -b feat/docker_ready --recursive git@github.com:Analytics4MD/A4MD-project-a4md.git a4md

RUN echo "ll python include dir--------------=========================="
RUN ls /usr/local/lib/python3.6/dist-packages/numpy/core/include

# Remove SSH keys
RUN rm -rf /root/.ssh/
RUN echo "---===== CLoging done"
#RUN python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
#RUN python3 -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"
RUN cd a4md && \
    git checkout feat/new_metric_names && \
    mkdir build && \
    cd build && \
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=../_install &&\
    make install

## Our software
#RUN apt update && \ 
#    apt install --no-install-recommends -y git && \
#    apt clean &&\
#    rm -rf /var/lib/apt/lists/* && \ 
#    conda install -y --only-deps cmake python=3.5 && \
#    conda clean -tipsy && \

##Hoomd
#RUN apt-get update && apt-get install -y --no-install-recommends git cmake && \
#    apt clean && \
#    rm -rf /var/lib/apt/lists/* && \
#    export HOOMD_TAG=v2.4.0-beta && \
#    git clone --recursive https://mikemhenry@bitbucket.org/cmelab/hoomd_blue.git && \
#    cd hoomd_blue && \
#    git checkout  $HOOMD_TAG \
#    && mkdir build \
#    && cd build \
#    export CXX="$(command -v g++)" && \
#    export CC="$(command -v gcc)" && \
#    cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` \
#              -DENABLE_CUDA=ON \
#              -DDISABLE_SQLITE=ON \
#              -DSINGLE_PRECISION=ON && \
#    make -j3 && \
#    make install
#
## epoxpy
#RUN pip install --no-cache-dir git+https://bitbucket.org@bitbucket.org/cmelab/epoxpy.git@mbuild_update 
#
## More things
#RUN pip install --no-cache-dir pytest pytest-cov coverage>=4.4 coverage>=4.4 coveralls PyYAML 
#
## MorphCT 
#RUN pip install --no-cache-dir git+https://bitbucket.org@bitbucket.org/cmelab/morphct.git@dev
#
## Rhaco
#RUN pip install --no-cache-dir git+https://bitbucket.org@bitbucket.org/cmelab/rhaco.git@dev
#
## mount points for filesystems on clusters
#RUN mkdir -p /nfs \
#    mkdir -p /oasis \
#    mkdir -p /scratch \
#    mkdir -p /work \
#    mkdir -p /projects \
#    mkdir -p /home1
#
## ORCA
#ENV ORCA_DIR="/opt/orca"
#ENV PATH="$ORCA_DIR/bin:$PATH"
#ENV LD_LIBRARY_PATH="$ORCA_DIR/lib:$LD_LIBRARY_PATH"
#
#COPY orca /opt/orca
#
#RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
#    imagemagick && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

