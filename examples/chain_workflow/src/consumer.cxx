#include <unistd.h>
#include "mpi.h"
#include "../../../src/a4md/dtl/include/dataspaces_reader.h"
#include "../../../src/a4md/common/include/md_stager.h"
#include "../../../src/a4md/common/include/timer.h"
#include "../../../src/a4md/retrieve/include/md_analyzer.h"
#include "../../../src/a4md/retrieve/include/md_retriever.h"

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Consummer::main()\n");
    if (argc != 6)
    {
        fprintf(stderr, "./consumer client_id group_id py_path py_func n_frames\n");
        return -1;
    }
    std::string py_path;
    std::string py_func;
    int client_id;
    int group_id;
    int n_frames;

    client_id = atoi(argv[1]);
    group_id = atoi(argv[2]);
    py_path = (char*)argv[3];
    py_func = (char*)argv[4];
    n_frames = atoi(argv[5]);
    // Number of chunks
    unsigned long int total_chunks = n_frames;// +1 for the call before simulation starts

    // Preprocess Python path
    std::size_t module_start = py_path.find_last_of("/");
    std::size_t module_end = py_path.rfind(".py");
    if (module_end == std::string::npos)
    {
    fprintf(stderr, "ERROR: Expecting a python module in the python script path argument.\n");
        return -1;
    }
    // get directory
    std::string py_dir = (std::string::npos==module_start)? std::string(".") : py_path.substr(0,module_start);
    // get file
    std::string py_name = py_path.substr(module_start+1, module_end-module_start-1);
    printf("Python directory : %s\n", py_dir.c_str());
    printf("Python script name : %s\n", py_name.c_str());
    printf("Python function: %s\n", py_func.c_str());
    
    ChunkReader* chunk_reader;
    chunk_reader = new DataSpacesReader(client_id, group_id, total_chunks, MPI_COMM_WORLD);

    ChunkWriter *chunk_writer = nullptr;
    // chunk_writer = new MDAnalyzer((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());

    ChunkStager *chunk_stager = new MDStager(chunk_reader, chunk_writer);
    int n_window_width = 1;
    Retriever *retriever = new MDRetriever(*chunk_stager, n_frames, n_window_width);
    
    // Main run
    TimeVar t_start = timeNow();
    retriever->run();
    DurationMilli md_retriever_time_ms = timeNow()-t_start;
    auto total_md_retriever_time_ms = md_retriever_time_ms.count();
    printf("total_retriever_time_ms : %f\n",total_md_retriever_time_ms);

    // Free memory
    delete retriever;
    delete chunk_stager;
    // delete chunk_writer;
    delete chunk_reader;

    MPI_Finalize();
    return 0;
}
