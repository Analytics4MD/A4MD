#include <unistd.h>
#include "mpi.h"
#include "decaf_reader.h"
#include <bredala/data_model/boost_macros.h>
#include "md_analyzer.h"
#include "md_retriever.h"
#include "md_stager.h"
#include "timer.h"

#define DTL_COLOR 0
#define NON_DTL_COLOR 1
#define ROOT 0

std::string analyzer_name = "md_analyzer";
std::string dtl_type = "decaf";

int main (int argc, const char** argv)
{
    if (argc != 5) 
    {
        fprintf(stderr, "ERROR: ./analyzer json_conf py_path py_func n_frames\n");
    }
    MPI_Init(NULL,NULL);
    
    MPI_Comm global_comm = MPI_COMM_WORLD; 
    int rank;
    MPI_Comm_rank(global_comm, &rank);

    int *appnum, present;
    MPI_Comm_get_attr(global_comm, MPI_APPNUM, &appnum, &present);
    MPI_Comm app_comm;
    MPI_Comm_split(global_comm, *appnum, rank, &app_comm);
    int app_rank;
    MPI_Comm_rank(app_comm, &app_rank);
    MPI_Comm_free(&app_comm);

    int color = (app_rank < 1) ? 0 : 1;
    MPI_Comm dtl_comm;
    MPI_Comm_split(global_comm, color, rank, &dtl_comm);

    if (app_rank == 0)
    {
        printf("---======== In Retriever::main()\n");
        std::string json_conf((char*)argv[1]);
        std::string py_path((char*)argv[2]);
        std::string py_func((char*)argv[3]);
        int n_frames = atoi(argv[4]);
        unsigned long int total_chunks = n_frames;// +1 for the call before simulation starts
        
        ChunkReader *chunk_reader;
        if (dtl_type == "decaf")
        {
            chunk_reader = new DecafReader(json_conf, total_chunks, dtl_comm);
        } 
        else
        {
            throw NotImplementedException("Reader type is not implemented");
        }

        ChunkWriter* chunk_writer;
        if(analyzer_name == "md_analyzer")
        {
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

            chunk_writer = new MDAnalyzer((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());
        }
        else
        {
            throw NotImplementedException("Analyzer type is not implemented");
        }

        ChunkStager *chunk_stager = new MDStager(chunk_reader, chunk_writer);
        int n_window_width = 1;
        Retriever *retriever = new MDRetriever(*chunk_stager, total_chunks, n_window_width);

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
    }
    MPI_Comm_free(&dtl_comm);
    MPI_Finalize();
    return 0;
}
