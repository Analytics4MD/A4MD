echo Building a4md dependency image

#docker build --build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" \
#                                      -t gclab/a4md . -f Dockerfile.build
docker build -t a4md_base . -f Dockerfile.build
