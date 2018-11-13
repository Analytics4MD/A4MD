#include "ingest.h"
#include "mpi.h"

int main (int argc, const char** argv)
{
  MPI_Init(NULL,NULL);
  // Ingest ingest;
  // ingest.run();

  MPI_Finalize();
  return 0;
}
