#include "ingest.h"

Ingest::Ingest()
{
    int nprocs;
    MPI_Comm gcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    gcomm = MPI_COMM_WORLD;
    int d_result = dspaces_init(nprocs,1,&gcomm, NULL);
    printf("In Ingest constructor: d_result %d\n",d_result);
    if (d_result != 0){
        if (d_result == -12)
            printf("Error initializing dataspaces. Could not find config file\n");
    }
}

Ingest::~Ingest()
{
    //Nothing for now
}

void Ingest::run()
{
    printf("In Ingest Run\n");
}
