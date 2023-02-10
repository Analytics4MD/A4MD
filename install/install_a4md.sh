cd ~/test/a4md
mkdir build
cd ~/test/a4md/build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/test/a4md-test -DDATASPACES_PREFIX=$HOME/dataspaces
make
make install
cd ~/test/a4md
pip install -e .

