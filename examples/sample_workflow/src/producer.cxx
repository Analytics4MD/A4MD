#include <string>

#include "mpi.h"
#ifdef DTL_DECAF
#include "decaf_writer.h"
#include <bredala/data_model/boost_macros.h>
#endif
#include "dataspaces_writer.h"
#include "pdb_chunker.h"
#include "md_stager.h"
#include "md_intermediator.h"
#include "md_generator.h"
#include "timer.h"



int main(int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Producer::main()\n");
    if (argc < 2)
    {
        fprintf(stderr, "./producer decaf json_conf py_path py_func n_frames n_atoms delay_ms\n");
        fprintf(stderr, "./producer dataspaces client_id group_id py_path py_func n_frames n_atoms delay_ms\n");
        return -1;
    }
    std::string dtl_type((char*)argv[1]);
#ifdef DTL_DECAF
    if ( (dtl_type.compare("decaf") != 0) && (dtl_type.compare("dataspaces") != 0) )
    {
        fprintf(stderr, "ERROR: DTL type must be decaf or dataspaces\n");
        return -1;
    }
#else 
    if (dtl_type.compare("dataspaces") != 0)  
    {
        fprintf(stderr, "ERROR: DTL type must be dataspaces\n");
        return -1;
    }  
#endif
    std::string json_conf;
    std::string py_path;
    std::string py_func;
    int client_id;
    int group_id;
    int n_frames;
    int n_atoms;
    int n_delay_ms;
    if (dtl_type.compare("decaf") == 0)
    {
        if (argc != 8)
        {
            fprintf(stderr, "ERROR: ./producer decaf json_conf py_path py_func n_frames n_atoms delay_ms\n"); 
            return -1;
        }
        // Input arguments
        json_conf = (char*)argv[2];
        py_path = (char*)argv[3];
        py_func = (char*)argv[4];
        n_frames = std::stoi(argv[5]);
        n_atoms = std::stoi(argv[6]);
        n_delay_ms = std::stoi(argv[7]);
    }

    if (dtl_type.compare("dataspaces") == 0)
    {
        if (argc != 9)
        {
            fprintf(stderr, "ERROR: ./producer dataspaces client_id group_id py_path py_func n_frames n_atoms delay_ms\n"); 
            return -1;
        }        
        // Input arguments
        client_id = std::stoi(argv[2]);
        group_id = std::stoi(argv[3]);
        py_path = (char*)argv[4];
        py_func = (char*)argv[5];
        n_frames = std::stoi(argv[6]);
        n_atoms = std::stoi(argv[7]);
        n_delay_ms = std::stoi(argv[8]);
    }

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
    
#if DTL_DECAF
    if (dtl_type.compare("decaf") == 0)
    {
        chunk_writer = new DecafWriter(json_conf, total_chunks, MPI_COMM_WORLD);
    }
#endif
    if (dtl_type.compare("dataspaces") == 0) 
    {
        chunk_writer = new DataSpacesWriter(client_id, group_id, total_chunks, MPI_COMM_WORLD);
    }

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
