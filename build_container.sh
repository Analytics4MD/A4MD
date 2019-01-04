echo Building a4md image

docker build --build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" \
                                      -t gclab/a4md . -f Dockerfile
