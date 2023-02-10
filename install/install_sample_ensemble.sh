echo 'Cloning and building sample workflow' >> installingProcess
cd
# git clone https://github.com/Analytics4MD/A4MD-sample-workflow.git sampleEnsemble
cd sampleEnsemble
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DA4MD_PREFIX=$HOME/test/a4md-test -DDATASPACES_PREFIX=$HOME/dataspaces
make
make install
cd ../bin/
