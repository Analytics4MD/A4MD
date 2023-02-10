#cd
#wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
#bash Anaconda3-5.3.1-Linux-x86_64.sh -b -p $HOME/anaconda3
. $HOME/anaconda3/etc/profile.d/conda.sh
echo 'Creating conda env' >> installingProcess
conda create -y -n a4md_conda_env numpy scipy
conda activate a4md_conda_env
conda install -y python=3.9.13=h2660328_0_cpython -c conda-forge
conda install -y -c conda-forge tbb tbb-devel cython sympy scikit-build
conda install -y -c conda-forge freud
pip install --upgrade MDAnalysis

