<h1 align="center">  
  A4MD
  <h4 align="center">
  <a href="https://analytics4md.org/"><img src="https://avatars.githubusercontent.com/u/32650548?s=200&v=4"/></a>
  </h4>
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#installation">Installation</a> •
  <a href="#publications">Publications</a> •
  <a href="#copyright-and-license">Copyright and License</a>
</p>

## About

The project's harnessed knowledge of molecular structures' transformations at runtime can be used to steer simulations to more promising areas of the simulation space, identify the data that should be written to congested parallel file systems, and index generated data for retrieval and post-simulation analysis. Supported by this knowledge, molecular dynamics workflows such as replica exchange simulations, Markov state models, and the string method with swarms of trajectories can be executed from the outside (i.e., without reengineering the molecular dynamics code) 

## Prerequisites

In order to use this package, your system should have the following installed:
- C++11
- cmake
- boost
- python3

(Optional) To use the built-in analysis library, it is required to install:
- mdtraj
- freud

## Dependencies

The framework is also built on top the following third-party libraries: 
- Dataspaces
- Decaf (optional) 

We also use Catch2 as a test framework.

## Installation

Here is the extensive installation instructions. Please make sure the all the prerequisites are satisfied before proceeding the following steps.
Make sure you are using ssh with GitHub and you have gcc compiler in your system. 

1. Clone the source code from this repository

```
git clone --recursive git@github.com:Analytics4MD/A4MD.git a4md
```

2. Build A4MD package 

```
cd a4md
. setup_deps.sh
```
The execution of previous script should create a folder called `a4md-test` in your home directory. This folder includes the binaries and examples to test A4MD.

### Run sample workflow
With all the installation process we have created a sample workflow, which consists of two consumers and two producers. To test this follow next steps

```
cd ~/a4md-test/examples/sample_workflow/
sh local.dspaces.prod_con.sh
```


### Additional data transport layer
 To build additional data transport layer based on Decaf, specify Decaf installation directory in the `install_a4md.sh` file as follows :

```
-Ddtl_decaf=on -DDECAF_PREFIX=${DECAF_ROOT}
```

## Publications

## Copyright and License

