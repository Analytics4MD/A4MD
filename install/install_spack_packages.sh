echo 'Installing modules with spack' >> installingProcess
spack env create a4md_spack_env
spack env activate a4md_spack_env

spack install gcc@5.5.0
spack load gcc@5.5.0
spack compiler add $(spack location -i gcc@5.5.0)

spack install mpich %gcc@5.5.0
spack load mpich %gcc@5.5.0

spack install boost %gcc@5.5.0 cxxstd=11 +iostreams +serialization
spack load boost %gcc@5.5.0 cxxstd=11 +iostreams +serialization

spack install cmake %gcc@5.5.0
spack load cmake %gcc@5.5.0
