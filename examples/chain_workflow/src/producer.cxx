#include <string>

#include "mpi.h"
#include "../../../src/a4md/dtl/include/dataspaces_writer.h"
#include "../../../src/a4md/ingest/include/pdb_chunker.h"
#include "../../../src/a4md/ingest/include/md_generator.h"
#include "../../../src/a4md/common/include/md_stager.h"
#include "../../../src/a4md/common/include/md_intermediator.h"
#include "../../../src/a4md/common/include/timer.h"

int main(int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    
    printf("---======== In Producer::main()\n");
    if (argc != 8)
    {
        fprintf(stderr, "./producer client_id group_id py_path py_func n_frames n_atoms delay_ms\n");
        return -1;
    }
    std::string py_path;
    std::string py_func;
    int client_id;
    int group_id;
    int n_frames;
    int n_atoms;
    int n_delay_ms;
  
    // Input arguments
    client_id = std::stoi(argv[1]);
    group_id = std::stoi(argv[2]);
    py_path = (char*)argv[3];
    py_func = (char*)argv[4];
    n_frames = std::stoi(argv[5]);
    n_atoms = std::stoi(argv[6]);
    n_delay_ms = std::stoi(argv[7]);
    // Number of chunks
    unsigned long int total_chunks = n_frames;

    // Preprocess Python script
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

    char *file_path = (char*)"";
    ChunkReader *chunk_reader = new PDBChunker((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str(), file_path, n_atoms, 0);

    ChunkWriter *chunk_writer;
    chunk_writer = new DataSpacesWriter(client_id, group_id, total_chunks, MPI_COMM_WORLD);

    ChunkStager *chunk_stager = new MDStager(chunk_reader, chunk_writer);
    Ingester *ingester = new MDGenerator(*chunk_stager, total_chunks, n_delay_ms);

    // Main run
    TimeVar t_start = timeNow();
    ingester->run();
    DurationMilli ingester_time_ms = timeNow() - t_start;
    auto total_ingester_time_ms = ingester_time_ms.count();
    printf("total_ingester_time_ms : %f\n", total_ingester_time_ms);
   
    // Free Memory
    delete ingester;
    delete chunk_stager;
    delete chunk_writer;
    delete chunk_reader;
    
    MPI_Finalize();
    return 0;
}
