#include "ingest.h"

Ingest::Ingest()
{
    int nprocs;
    MPI_Comm gcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    //dspaces_init(nprocs,1,&gcomm, NULL);
}

Ingest::~Ingest()
{
    //Nothing for now
}

void Ingest::run()
{
    printf("In Ingest Run\n");
}
