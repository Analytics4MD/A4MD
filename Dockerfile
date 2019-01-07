FROM globalcomputinglab/a4md_base:latest

WORKDIR /app
ADD _install/ /app
RUN echo "export PATH='/app/bin/:$PATH'" >> ~/.bashrc
