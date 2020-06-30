// -*- c++ -*-
//
// world48
// Todd Gamblin, tgamblinWllnl.gov
//
// This file is an example of how to use wrap.py to fool an application
// into thinking it is running on a different communicator from
// MPI_Comm_world.
//
// This was originally intended to allow applications on Blue Gene/Q to
// run with 48 MPI processes per node, rather than just the power of 2
// that IBM provides settings for.  The MPI_Init wrapper here will
// split MPI_Comm_world into 2 groups: one for the first 48 out of every
// 64 ranks and one for the last 16.  The last 16 ranks of every 64 just
// call MPI_Finalize and exit normally inside of MPI_Init.  The rest of
// the ranks continue to execute the rest of the application, thinking
// that the world is only 1/4 as big as the real MPI_COMM_WORLD.
//
// To build:
//    wrap.py world48.w > world48.C
//    mpicc -c world48.C
//    ar cr libworld48.a world48.o
//    ranlib libworld48.a
//
// Link your application with libworld48.a, or build it as a shared lib
// and LD_PRELOAD it to try out this tool.
//
#include <mpi.h>

// This is a communicator that will contain the first 48 out of
// every 64 ranks in the application.
static MPI_Comm app_comm;

// This function modifies its parameter by swapping it with world48
// if it is MPI_COMM_WORLD.
inline void swap_world(MPI_Comm& world) {
   if (world == MPI_COMM_WORLD) {
      world = app_comm;
   }
}

// MPI_Init does all the communicator setup
//
{{fn func MPI_Init}}{
   // First call PMPI_Init()
   {{callfn}}
   printf("Wrap: MPI_Init\n");
   int rank;
   PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
   printf("Wrap: rank = %d\n", rank);

   int *appnum, present;
   PMPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum, &present);
   PMPI_Comm_split(MPI_COMM_WORLD, *appnum, rank, &app_comm);
   //int keep = (rank % 6 < 4) ? 1: 0;
   //PMPI_Comm_split(MPI_COMM_WORLD,  keep, rank, &app_comm);
}{{endfn}}

// This generates interceptors that will catch every MPI routine
// *except* MPI_Init.  The interceptors just make sure that if
// they are called with an argument of type MPI_Comm that has a
// value of MPI_COMM_WORLD, they switch it with world48.
{{fnall func MPI_Init}}{
   {{apply_to_type MPI_Comm swap_world}}
   {{callfn}}
}{{endfnall}}
