FROM globalcomputinglab/a4md_base_openmpi_2.1.3:latest

WORKDIR /app
ADD _install/ /app
ENV PATH="/app/bin/:${PATH}"
