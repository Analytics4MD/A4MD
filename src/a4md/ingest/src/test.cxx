#include "../include/test.h"
#include <stdio.h>

DataSpacesWriter1::DataSpacesWriter1(string var_name)
{
    m_gcomm = MPI_COMM_WORLD;
    printf("Initialized DataSpacesWriter1\n");
}
