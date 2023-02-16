#!/usr/bin/env bash
install_dir=$1
mkdir -p ${install_dir}
git clone --recursive git@github.com:Analytics4MD/A4MD.git ${install_dir}
