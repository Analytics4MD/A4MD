#include "mpi.h"
#include "dataspaces.h"
#include <string>
using namespace std;

class DataSpacesWriter1
{
    private:
        MPI_Comm m_gcomm;
    public:
        DataSpacesWriter1(string var_name);
};
