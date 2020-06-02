#include <unistd.h>
#include "mpi.h"
#ifdef DTL_DECAF
#include "decaf_reader.h"
#include <bredala/data_model/boost_macros.h>
#endif
#include "dataspaces_reader.h"
#include "md_stager.h"
#include "md_analyzer.h"
#include "md_retriever.h"
#include "timer.h"

std::string dtl_type = "dataspaces";

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Consummer::main()\n");
    if (argc < 2)
    {
        fprintf(stderr, "./consumer decaf json_conf py_path py_func n_frames\n");
        fprintf(stderr, "./consumer dataspaces client_id group_id py_path py_func n_frames\n");
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
    if (dtl_type.compare("decaf") == 0)
    {
        if (argc != 6)
        {
            fprintf(stderr, "ERROR: ./consumer decaf json_conf py_path py_func n_frames\n"); 
            return -1;
        }
        // Input arguments
        json_conf = (char*)argv[2];
        py_path = (char*)argv[3];
        py_func = (char*)argv[4];
        n_frames = atoi(argv[5]);   
    }

    if (dtl_type.compare("dataspaces") == 0)
    {
        if (argc != 7)
        {
            fprintf(stderr, "ERROR: ./consumer dataspaces client_id group_id py_path py_func n_frames\n"); 
            return -1;
        }        
        // Input arguments
        client_id = atoi(argv[2]);
        group_id = atoi(argv[3]);
        py_path = (char*)argv[4];
        py_func = (char*)argv[5];
        n_frames = atoi(argv[6]);
    }

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
#ifdef DTL_DECAF
    if (dtl_type.compare("decaf") == 0)
    {
        chunk_reader = new DecafReader(json_conf, total_chunks, MPI_COMM_WORLD);
    }
#endif
    if (dtl_type.compare("dataspaces") == 0)
    {
        chunk_reader = new DataSpacesReader(client_id, group_id, total_chunks, MPI_COMM_WORLD);
    }

    ChunkWriter *chunk_writer = new MDAnalyzer((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());

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
    delete chunk_writer;
    delete chunk_reader;

    MPI_Finalize();
    return 0;
}
