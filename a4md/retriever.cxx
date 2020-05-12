#include "mpi.h"
#include "dataspaces_reader.h"
#include "md_analyzer.h"
#include "md_retriever.h"
#include <unistd.h>
#include "timer.h"

ChunkAnalyzer* analyzer_factory(int argc, const char** argv)
{
    

    std::string analyzer_name = "md_analyzer";
    std::string reader_type = "dataspaces";
    std::string var_name = "test_var";

    ChunkAnalyzer* chunk_analyzer;
    ChunkReader* chunk_reader;
    int n_frames = atoi(argv[3]);
    int n_analysis_stride = 1;
    unsigned long int total_chunks = n_frames;// +1 for the call before simulation starts
    if (reader_type == "dataspaces")
    {
        printf("---======== Initializing dataspaces reader\n");
        Chunker * chunker = new DataSpacesReader(2,(char*)var_name.c_str(), total_chunks, MPI_COMM_WORLD);
        printf("---======== Initialized dataspaces reader\n");
        chunk_reader = new ChunkReader(* chunker);
        printf("---======== Initialized chunkreader\n");
    }
    else
    {
        throw NotImplementedException("Reader type is not implemented");
    }

    if(analyzer_name == "md_analyzer")
    {
        std::string name((char*)argv[1]);
        std::string func((char*)argv[2]);
        PyRunner* py_runner = new PyRunner((char*)name.c_str(),(char*)func.c_str());
        chunk_analyzer = new MDAnalyzer(*chunk_reader, *py_runner);
        printf("---======== Initialized md analyzer\n");
    }
    else
    {
        throw NotImplementedException("Analyzer type is not implemented");
    }

    return chunk_analyzer;
}

Retriever* retriever_factory (int argc, const char** argv)
{
    ChunkAnalyzer * analyzer = analyzer_factory(argc, argv);
    int n_frames = atoi(argv[3]);
    int n_window_width = 1;
    
    printf("Recieved n_frames = %i from user in Retriever\n",n_frames);
    Retriever * retriever = new MDRetriever(* analyzer, n_frames, n_window_width);
    return retriever;
}

int main (int argc, const char** argv)
{
    if (argc != 4)
        printf("ERROR: Expecting 4 command line arguments 1) python module name 2) function name 3) n_frames\n");
    MPI_Init(NULL,NULL);
    printf("---======== In Retriever::main()\n");

    TimeVar t_start = timeNow();
    Retriever * retriever = retriever_factory(argc,argv);
    retriever->run();
    DurationMilli md_retriever_time_ms = timeNow()-t_start;
    auto total_md_retriever_time_ms = md_retriever_time_ms.count();
    printf("total_retriever_time_ms : %f\n",total_md_retriever_time_ms);

    MPI_Finalize();
    return 0;
}
