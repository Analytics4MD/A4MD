#include <unistd.h>
#include "mpi.h"
#include "dataspaces_reader.h"
#include "md_analyzer.h"
#include "md_retriever.h"
#include "timer.h"

std::string analyzer_name = "md_analyzer";
std::string reader_type = "dataspaces";
std::string var_name = "test_var";

int main (int argc, const char** argv)
{
    if (argc != 4) 
    {
        fprintf(stderr, "ERROR: Expecting 4 command line arguments 1) python module name 2) function name 3) n_frames\n");
    }
    MPI_Init(NULL,NULL);
    printf("---======== In Retriever::main()\n");
    
    Chunker *chunker;
    if (reader_type == "dataspaces")
    {
        int n_frames = atoi(argv[3]);
        int n_analysis_stride = 1;
        unsigned long int total_chunks = n_frames;// +1 for the call before simulation starts
        chunker = new DataSpacesReader((char*)var_name.c_str(), total_chunks, MPI_COMM_WORLD);
    }
    else
    {
        throw NotImplementedException("Reader type is not implemented");
    }
    ChunkReader *chunk_reader = new ChunkReader(* chunker);

    PyRunner *py_runner;
    ChunkAnalyzer *chunk_analyzer;
    Retriever *retriever;
    if(analyzer_name == "md_analyzer")
    {
        std::string py_name((char*)argv[1]);
        std::string py_func((char*)argv[2]);
        py_runner = new PyRunner((char*)py_name.c_str(),(char*)py_func.c_str());
        chunk_analyzer = new MDAnalyzer(*chunk_reader, *py_runner);
        int n_frames = atoi(argv[3]);
        int n_window_width = 1;
        printf("Received n_frames = %i from user in Retriever\n",n_frames);
        retriever = new MDRetriever(*chunk_analyzer, n_frames, n_window_width);
    }
    else
    {
        throw NotImplementedException("Analyzer type is not implemented");
    }

    TimeVar t_start = timeNow();
    retriever->run();
    DurationMilli md_retriever_time_ms = timeNow()-t_start;
    auto total_md_retriever_time_ms = md_retriever_time_ms.count();
    printf("total_retriever_time_ms : %f\n",total_md_retriever_time_ms);

    // Free memory
    delete retriever;
    delete chunk_analyzer;
    delete py_runner;
    delete chunk_reader;
    delete chunker;

    MPI_Finalize();
    return 0;
}
