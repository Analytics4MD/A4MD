FROM globalcomputinglab/a4md_base:latest

WORKDIR /app
ADD _install/ /app
ENV PATH="/app/bin/:${PATH}"
