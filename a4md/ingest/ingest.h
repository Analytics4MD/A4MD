#ifndef __INGEST_H__
#define __INGEST_H__
#include <stdio.h>
#include "dataspaces.h"
#include "mpi.h"

class Ingest
{
    public:
        Ingest();
        ~Ingest();
        void run();
};

#endif
