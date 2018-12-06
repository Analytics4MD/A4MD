#include "retrieve.h"
#include "mpi.h"
#include "dataspaces_reader.h"
#include <unistd.h>


void analyze(Retrieve& retrieve, int step, int argc, const char** argv, ChunkArray chunk_ary)
{

    auto chunks = chunk_ary.get_chunks();
    for (auto tempchunk:chunks)
    {
        PLMDChunk *chunk = dynamic_cast<PLMDChunk *>(tempchunk);
        //printf("Printing typecasted chunk\n");
        //chunk->print();
        POS_VEC positions = chunk->get_positions();
        auto types_vector = chunk->get_types();
        int* types = types_vector.data();

        //for (int i=0;i< types_vector.size(); i++)
        //    printf("type: %i ",types[i]);
        //printf("\n----=======Positions\n");
        //for (auto pos:positions)
        //    printf("%f %f %f \n",std::get<0>(pos), std::get<1>(pos),std::get<2>(pos));
        //printf("----=======Positions end\n");
        double lx, ly, lz, xy, xz, yz; //xy, xz, yz are tilt factors 
        lx = chunk->get_box_lx();
        ly = chunk->get_box_ly();
        lz = chunk->get_box_lz();
        xy = chunk->get_box_xy(); // 0 for orthorhombic
        xz = chunk->get_box_xz(); // 0 for orthorhombic
        yz = chunk->get_box_yz(); // 0 for orthorhombic
      
        char* name = (char*)argv[1];
        char* func = (char*)argv[2];
        retrieve.analyze_frame(name,
                               func,
                               types,
                               positions,
                               lx,
                               ly,
                               lz,
                               xy,
                               xz,
                               yz,
                               step);

    }
}

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    
    Retrieve retrieve;
    std::string var_name = "test_var";
    DataSpacesReader dataspaces_reader_ptr = DataSpacesReader((char*)var_name.c_str());

    //printf("Waiting 5 seconds before Read 1\n");
    //sleep(5);
    int total_time = atoi(argv[3]);
    int dump_interval = atoi(argv[4]);
    printf("Recieved total time = %i from user in Retriever\n",total_time);
    for (int timestep=0;timestep<=total_time;timestep+=dump_interval) // NOTE: we get timestep 0 to the total time from lammps
    {
        auto chunk_array = dataspaces_reader_ptr.get_chunks(timestep);
        //chunk_array.print();
        printf("Analyzing time step %i\n",timestep);
        analyze(retrieve, timestep, argc, argv, chunk_array);
    }
    //chunk_array = dataspaces_reader_ptr.get_chunks(101);
    //chunk_array.print();
    //printf("Analyzing 2\n");
    //analyze(retrieve, argc, argv, chunk_array);

    MPI_Finalize();
    return 0;//retrieve.call_py(argc, argv);
}
