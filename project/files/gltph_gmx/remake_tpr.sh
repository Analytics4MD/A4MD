#! /bin/bash

gmx_mpi grompp -f bench.mdp -p ../../files/gltph_gmx/top/topol.top -n index.ndx -c start_conf.gro -t start_state.cpt -r start_conf.gro -o topol.tpr