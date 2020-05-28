#include <unistd.h>
#include "mpi.h"
#include "dataspaces_reader.h"
// #include "md_runner.h"
#include "md_analyzer.h"
#include "md_retriever.h"
#include "md_stager.h"
#include "timer.h"

std::string analyzer_name = "md_analyzer";
std::string reader_type = "dataspaces";

int main (int argc, const char** argv)
{
    if (argc != 6) 
    {
        fprintf(stderr, "ERROR: ./analyzer client_id group_id py_path py_func n_frames\n");
    }
    MPI_Init(NULL,NULL);
    printf("---======== In Retriever::main()\n");

    int n_frames = atoi(argv[5]);
    unsigned long int total_chunks = n_frames;// +1 for the call before simulation starts
    
    ChunkReader *chunk_reader;
    if (reader_type == "dataspaces")
    {
        int client_id = atoi(argv[1]);
        int group_id = atoi(argv[2]);
        int n_analysis_stride = 1;
        chunk_reader = new DataSpacesReader(2, 1, total_chunks, MPI_COMM_WORLD);
    }
    else
    {
        throw NotImplementedException("Reader type is not implemented");
    }

    // PyRunner *py_runner;
    // ChunkAnalyzer *chunk_analyzer;
    ChunkWriter* chunk_writer;
    // Retriever *retriever;
    if(analyzer_name == "md_analyzer")
    {
        std::string py_path((char*)argv[3]);
        std::string py_func((char*)argv[4]);
        std::size_t module_start = py_path.find_last_of("/");
        std::size_t module_end = py_path.rfind(".py");
        if (module_end == std::string::npos)
        {
        fprintf(stderr, "ERROR: Expecting a python module in the py_path argument.\n");
            return -1;
        }
        // get directory
        std::string py_dir = (std::string::npos==module_start)? std::string(".") : py_path.substr(0,module_start);
        // get file
        std::string py_name = py_path.substr(module_start+1, module_end-module_start-1);
        printf("Python directory : %s\n", py_dir.c_str());
        printf("Python script name : %s\n", py_name.c_str());
        printf("Python function: %s\n", py_func.c_str());

        // py_runner = new MDRunner((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());
        // chunk_analyzer = new MDAnalyzer(*chunk_reader, *py_runner);
        chunk_writer = new MDAnalyzer((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());

        // int n_frames = atoi(argv[5]);
        
        // printf("Received n_frames = %i from user in Retriever\n",n_frames);
        // retriever = new MDRetriever(*chunk_analyzer, n_frames, n_window_width);
    }
    else
    {
        throw NotImplementedException("Analyzer type is not implemented");
    }

    ChunkStager *chunk_stager = new MDStager(*chunk_reader, *chunk_writer);
    int n_window_width = 1;
    Retriever *retriever = new MDRetriever(*chunk_stager, total_chunks, n_window_width);

    TimeVar t_start = timeNow();
    retriever->run();
    DurationMilli md_retriever_time_ms = timeNow()-t_start;
    auto total_md_retriever_time_ms = md_retriever_time_ms.count();
    printf("total_retriever_time_ms : %f\n",total_md_retriever_time_ms);

    // Free memory
    delete retriever;
    // delete chunk_analyzer;
    delete chunk_stager;
    // delete py_runner;
    delete chunk_writer;
    delete chunk_reader;

    MPI_Finalize();
    return 0;
}
