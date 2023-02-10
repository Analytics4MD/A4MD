echo 'Building DSpaces' >> installingProcess
# Build and install dataspaces into $HOME/dataspaces
cd ~/test/a4md/src/a4md/extern/dataspaces
./autogen.sh
CC=$(which mpicc) CXX=$(which mpicxx) FC=$(which mpifort) CFLAGS=-fPIC ./configure --enable-shmem --enable-dart-tcp --prefix=$HOME/dataspaces
make
make install

