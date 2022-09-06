#include <unistd.h>
#include "mpi.h"
#ifdef DTL_DECAF
#include "decaf_reader.h"
#include "decaf_writer.h"
#include <bredala/data_model/boost_macros.h>
#endif
#include "../../../src/a4md/dtl/include/dataspaces_writer.h"
#include "../../../src/a4md/dtl/include/dataspaces_reader.h"
#include "../../../src/a4md/common/include/md_intermediator.h"
#include "../../../src/a4md/common/include/md_stager.h"
#include "../../../src/a4md/common/include/timer.h"
#include "../../../src/a4md/ingest/include/md_generator.h"

#define DTL_COLOR 0
#define NON_DTL_COLOR 1
#define ROOT 0

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    int rank, app_rank;
    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm app_comm, dtl_comm;
    int color;
    int *appnum, present;
    MPI_Comm_rank(global_comm, &rank);
    MPI_Comm_get_attr(global_comm, MPI_APPNUM, &appnum, &present);
    MPI_Comm_split(global_comm, *appnum, rank, &app_comm);
    MPI_Comm_rank(app_comm, &app_rank);
    MPI_Comm_free(&app_comm);
    if (app_rank == ROOT) 
    {
        color = DTL_COLOR;
    } 
    else
    {
        color = NON_DTL_COLOR; 
    }
    MPI_Comm_split(global_comm, color, rank, &dtl_comm);

    if (app_rank == ROOT)
    {
        printf("---======== In Intermediator::main()\n");
        if (argc < 2)
        {
            fprintf(stderr, "./intermediator decaf json_conf py_path py_func n_frames\n");
            fprintf(stderr, "./intermediator dataspaces reader_client_id reader_group_id writer_client_id writer_group_id py_path py_func n_frames\n");
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
        int n_frames;
        int reader_client_id;
        int reader_group_id;
        int writer_client_id;
        int writer_group_id;

        if (dtl_type.compare("decaf") == 0)
        {
            if (argc != 6)
            {
                fprintf(stderr, "ERROR: ./intermediator decaf json_conf py_path py_func n_frames\n"); 
                return -1;
            }
            // Input arguments
            json_conf = (char*)argv[2];
            py_path = (char*)argv[3];
            py_func = (char*)argv[4];
            n_frames = std::stoi(argv[5]);  
        }

        if (dtl_type.compare("dataspaces") == 0)
        {
            if (argc != 9)
            {
                fprintf(stderr, "ERROR: ./intermediator dataspaces reader_client_id reader_group_id writer_client_id writer_group_id py_path py_func n_frames\n"); 
                return -1;
            }        
            // Input arguments
            reader_client_id = std::stoi(argv[2]);
            reader_group_id = std::stoi(argv[3]);
            writer_client_id = std::stoi(argv[4]);
            writer_group_id = std::stoi(argv[5]);
            py_path = (char*)argv[6];
            py_func = (char*)argv[7];
            n_frames = std::stoi(argv[8]);
        }
      
        // Number of chunks
        unsigned long int total_chunks = n_frames;// +1 for the call before simulation starts

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

        ChunkReader* chunk_reader;
        ChunkWriter* chunk_writer;
#ifdef DTL_DECAF
        decaf::Decaf *decaf;
        if (dtl_type.compare("decaf") == 0)
        {
            Workflow workflow;
            Workflow::make_wflow_from_json(workflow, json_conf.c_str());
            printf("Initializing decaf\n");
            decaf = new decaf::Decaf(dtl_comm, workflow);
            chunk_writer = new DecafWriter(decaf, total_chunks, dtl_comm);
            chunk_reader = new DecafReader(decaf, total_chunks, dtl_comm);
        }
#endif
        if (dtl_type.compare("dataspaces") == 0)
        {        
            chunk_writer = new DataSpacesWriter(writer_client_id, writer_group_id, total_chunks, dtl_comm);
            chunk_reader = new DataSpacesReader(reader_client_id, reader_group_id, total_chunks, dtl_comm);
        }

        ChunkOperator *chunk_operator = new MDIntermediator((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());

        ChunkStager *chunk_stager = new MDStager(chunk_reader, chunk_operator, chunk_writer);
        Ingester *ingester = new MDGenerator(*chunk_stager, total_chunks, 0);
        
        // Main run
        TimeVar t_start = timeNow();
        ingester->run();
        DurationMilli intermediator_time_ms = timeNow() - t_start;
        auto total_intermediator_time_ms = intermediator_time_ms.count();
        printf("total_intermediator_time_ms : %f\n", total_intermediator_time_ms);

        // Free memory
        delete ingester;
        delete chunk_stager;
        delete chunk_operator;
        delete chunk_writer;
        delete chunk_reader;

#ifdef DTL_DECAF
        if (dtl_type.compare("decaf") == 0)
        {
            printf("Terminating decaf\n");
            decaf->terminate();
            delete decaf;
        }
#endif
    }
    
    MPI_Comm_free(&dtl_comm);
    MPI_Finalize();
    return 0;
}
