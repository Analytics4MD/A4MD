#include <unistd.h>
#include "mpi.h"
#include "../../../src/a4md/dtl/include/dataspaces_writer.h"
#include "../../../src/a4md/dtl/include/dataspaces_reader.h"
#include "../../../src/a4md/common/include/md_intermediator.h"
#include "../../../src/a4md/common/include/md_stager.h"
#include "../../../src/a4md/ingest/include/md_generator.h"
#include "../../../src/a4md/common/include/cv_runner.h"
#include "../../../src/a4md/ingest/include/cv_buffer.h"
#include "../../../src/a4md/common/include/timer.h"

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Intermediator::main()\n");
    if (argc < 9)
    {
        fprintf(stderr, "./intermediator reader_client_id reader_group_id writer_client_id writer_group_id py_path py_func n_frames\n");
        return -1;
    }
    std::string py_path;
    std::string py_func;
    int n_frames;
    int reader_client_id;
    int reader_group_id;
    int writer_client_id;
    int writer_group_id;
    int interval;

    // Input arguments
    reader_client_id = std::stoi(argv[1]);
    reader_group_id = std::stoi(argv[2]);
    writer_client_id = std::stoi(argv[3]);
    writer_group_id = std::stoi(argv[4]);
    py_path = (char*)argv[5];
    py_func = (char*)argv[6];
    n_frames = std::stoi(argv[7]);
    interval = std::stoi(argv[8]);
  
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
    chunk_writer = new DataSpacesWriter(writer_client_id, writer_group_id, total_chunks, MPI_COMM_WORLD);
    chunk_reader = new DataSpacesReader(reader_client_id, reader_group_id, total_chunks, MPI_COMM_WORLD);

    PyRunner* py_runner = new CVRunner((char*)py_name.c_str(), (char*)py_func.c_str(), (char*)py_dir.c_str());
    ChunkOperator *chunk_operator_1 = new MDIntermediator(py_runner);
    ChunkOperator *chunk_operator_2 = new CVBuffer(interval);

    ChunkStager *chunk_stager = new MDStager(chunk_reader, std::vector<ChunkOperator*>{chunk_operator_1, chunk_operator_2}, chunk_writer);
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
    delete chunk_operator_1;
    delete chunk_operator_2;
    delete chunk_writer;
    delete chunk_reader;

    MPI_Finalize();
    return 0;
}
