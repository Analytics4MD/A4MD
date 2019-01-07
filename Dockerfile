FROM globalcomputinglab/a4md_base:latest

WORKDIR /app
ADD _install/ /app
RUN export PATH="/app/bin/:$PATH"
