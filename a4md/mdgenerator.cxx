#include "mpi.h"
#include "py_runner.h"
#include "chunk.h"
#include "pdb_chunker.h"
#include "dataspaces_writer.h"
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <thread>


int main(int argc, const char** argv)
{
    MPI_Init(NULL, NULL);
    if (argc != 7)
    {
        fprintf(stderr, "ERROR: ./mdgenerator py_path py_func file_path n_frames n_atoms delay_ms\n"); 
        return -1;
    }
    printf("----===== Initializing PDBChunker =====----\n");
    std::string py_path((char*)argv[1]);
    std::string py_func((char*)argv[2]);
    std::string file_path((char*)argv[3]);
    int n_frames = std::stoi(argv[4]);
    int n_atoms = std::stoi(argv[5]);
    int n_delay_ms = std::stoi(argv[6]);
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

    PyRunner* py_runner = new PyRunner((char*)py_name.c_str(), 
                                       (char*)py_func.c_str(),
                                       (char*)py_path.c_str());
    PDBChunker* pdb_chunker = new PDBChunker((*py_runner), (char*)file_path.c_str());
    printf("----===== Initialized PDBChunker =====----\n");
    
    printf("----===== Initializing DataSpaces Writer ====----\n");
    char* var_name = "test_var";
    unsigned long int total_chunks = n_frames;
    DataSpacesWriter *dataspaces_writer_ptr = new DataSpacesWriter(var_name, total_chunks, MPI_COMM_WORLD);
    printf("----===== Initialized DataSpaces Writer ====----\n");

    TimeVar t_start = timeNow();
    for (auto step = 0; step < total_chunks; step++) 
    {
        std::cout << "Iteration : " << step << std::endl;
        if (pdb_chunker->extract_chunk(n_atoms) == 0) 
        {
            std::vector<Chunk*> chunks = pdb_chunker->get_chunks(1);
            printf("----===== Writing Chunk %i to DataSpaces START====----\n",step);
            dataspaces_writer_ptr->write_chunks(chunks);
            MDChunk *chunk = dynamic_cast<MDChunk *>(chunks.front());
            delete chunk;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(n_delay_ms));
    }
    DurationMilli md_generator_time_ms = timeNow() - t_start;
    auto total_md_generator_time_ms = md_generator_time_ms.count();
    printf("total_md_generator_time_ms : %f\n", total_md_generator_time_ms);
    // ToDo: check if actually finish writing to DataSpaces by adding dspaces_put_sync or --enable-sync-msg

    // ToDo: Free memory
    //delete dataspaces_writer_ptr;
    //delete pdb_chunker;
    //delete py_runner;
    MPI_Finalize();
    return 0;
}
