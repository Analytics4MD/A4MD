#include "retrieve.h"
#include "mpi.h"
#include "dataspaces_reader.h"
#include <unistd.h>


void analyze(Retrieve& retrieve, int argc, const char** argv, ChunkArray chunk_ary)
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
        double x_low, y_low, z_low,x_high, y_high, z_high;
        x_low=y_low=z_low=0.0;
        x_high=y_high=z_high=1.0;
        //printf("x_low %f\n",x_low);
        //printf("x_high %f\n",x_high);

        //printf("y_low %f\n",y_low);
        //printf("y_high %f\n",y_high);

        //printf("z_low %f\n",z_low);
        //printf("z_high %f\n",z_high);
        char* name = (char*)argv[1];
        char* func = (char*)argv[2];
        retrieve.analyze_frame(name,
                               func,
                               types,
                               positions,
                               x_low,
                               x_high,
                               y_low,
                               y_high,
                               z_low,
                               z_high,
                               0);

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
    auto chunk_array = dataspaces_reader_ptr.get_chunks(100);
    chunk_array.print();
    printf("Analyzing 1\n");
    analyze(retrieve, argc, argv, chunk_array);

    chunk_array = dataspaces_reader_ptr.get_chunks(101);
    chunk_array.print();
    printf("Analyzing 2\n");
    analyze(retrieve, argc, argv, chunk_array);

    MPI_Finalize();
    return 0;//retrieve.call_py(argc, argv);
}
