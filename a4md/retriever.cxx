#include "retrieve.h"
#include "mpi.h"
#include "dataspaces_reader.h"
#include "voronoi_analyzer.h"
#include "md_retriever.h"
#include <unistd.h>


void analyze(Retrieve& retrieve, int step, int argc, const char** argv, ChunkArray chunk_ary)
{

/*    auto chunks = chunk_ary.get_chunks();
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

    }*/
}

ChunkAnalyzer* analyzer_factory(int argc, const char** argv)
{
    int total_time = atoi(argv[3]);
    int dump_interval = atoi(argv[4]);
    printf("Recieved total time = %i from user in Retriever\n",total_time);

    std::string analyzer_name = "voronoi_analyzer";
    std::string reader_type = "dataspaces";
    std::string var_name = "test_var";

    //ChunkAnalyzer* chunk_analyzer;
    //ChunkReader* chunk_reader;
    //switch (reader_type)
    //{
    //    case "dataspaces":
    printf("---======== Initializing dataspaces reader\n");

    Chunker * chunker = new DataSpacesReader((char*)var_name.c_str());
    printf("---======== Initialized dataspaces reader\n");
    ChunkReader * chunk_reader = new ChunkReader(* chunker);
    printf("---======== Initialized chunkreader\n");

    //        break;
    //    default:
    //        throw NotImplementedException("Reader type %s is not implemented\n",reader_type);
    //}

    //switch (analyzer_name)
    //{
    //    case "voronoi_analyzer":
    std::string name((char*)argv[1]);
    std::string func((char*)argv[2]);

    ChunkAnalyzer * chunk_analyzer = new VoronoiAnalyzer(* chunk_reader, name, func);
    printf("---======== Initialized voronoi analyzer\n");

    //        break;
    //    default:
    //        throw NotImplementedException("Analyzer of type %s is not implemented\n",analyzer_name);
    //}

    return chunk_analyzer;
}

Retriever* retriever_factory (int argc, const char** argv)
{
    ChunkAnalyzer * analyzer = analyzer_factory(argc, argv);
    int n_steps = 2;
    int n_stride = 1;
    int n_analysis_stride = 1;
    Retriever * retriever = new MDRetriever(* analyzer, n_steps, n_stride, n_analysis_stride);
    return retriever;
}

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Retriever::main()\n");

    Retriever * retriever = retriever_factory(argc,argv);
    retriever->run();
    
    MPI_Finalize();
    return 0;
}
