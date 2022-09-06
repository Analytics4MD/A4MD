#include <string>

#include "mpi.h"
#include "ingest/include/pdb_chunker.h"
#include "dtl/include/dataspaces_writer.h"
#include "common/include/md_stager.h"
#include "ingest/include/md_generator.h"
#include "common/include/timer.h"

std::string reader_type = "pdb";
std::string writer_type = "dataspaces";

int main(int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Generator::main()\n");
    if (argc != 9)
    {
        fprintf(stderr, "ERROR: ./generator client_id group_id py_path py_func file_path n_frames n_atoms delay_ms\n"); 
        return -1;
    }
    int client_id = atoi(argv[1]);
    int group_id = atoi(argv[2]);
    std::string py_path((char*)argv[3]);
    std::string py_func((char*)argv[4]);
    std::string file_path((char*)argv[5]);
    int n_frames = std::stoi(argv[6]);
    int n_atoms = std::stoi(argv[7]);
    int n_delay_ms = std::stoi(argv[8]);
    unsigned long int total_chunks = n_frames;
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

    ChunkReader *chunk_reader;
    if (reader_type == "pdb")
    {
        chunk_reader = new PDBChunker((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str(), (char*)file_path.c_str(), n_atoms, 0);
    }
    else 
    {
        throw NotImplementedException("Reader type is not implemented\n");
    }

    ChunkWriter *chunk_writer;
    if (writer_type == "dataspaces") 
    {
        chunk_writer = new DataSpacesWriter(client_id, group_id, total_chunks, MPI_COMM_WORLD);
    }
    else 
    {
        throw NotImplementedException("Writer type is not implemented\n");
    }

    ChunkStager *chunk_stager = new MDStager(chunk_reader, chunk_writer);
    Ingester *ingester = new MDGenerator(*chunk_stager, total_chunks, n_delay_ms);

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

