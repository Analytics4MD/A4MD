#include <string>

#include "mpi.h"
#include "py_runner.h"
#include "pdb_chunker.h"
#include "dataspaces_writer.h"
#include "md_stager.h"
#include "md_generator.h"
#include "timer.h"

std::string reader_type = "pdb";
std::string writer_type = "dataspaces";

int main(int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Generator::main()\n");
    if (argc != 7)
    {
        fprintf(stderr, "ERROR: ./generator py_path py_func file_path n_frames n_atoms delay_ms\n"); 
        return -1;
    }
    std::string py_path((char*)argv[1]);
    std::string py_func((char*)argv[2]);
    std::string file_path((char*)argv[3]);
    int n_frames = std::stoi(argv[4]);
    int n_atoms = std::stoi(argv[5]);
    int n_delay_ms = std::stoi(argv[6]);
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

    IMSReader *ims_reader;
    PyRunner *py_runner;
    if (reader_type == "pdb")
    {
        py_runner = new PyRunner((char*)py_name.c_str(), 
                                           (char*)py_func.c_str(),
                                           (char*)py_dir.c_str());
        ims_reader = new PDBChunker((*py_runner), (char*)file_path.c_str(), 0, n_atoms);
    }
    else 
    {
        throw NotImplementedException("Reader type is not implemented\n");
    }
    ChunkReader *chunk_reader = new ChunkReader(*ims_reader);

    IMSWriter *ims_writer;
    if (writer_type == "dataspaces") 
    {
        ims_writer = new DataSpacesWriter(1, 1, total_chunks, MPI_COMM_WORLD);
    }
    else 
    {
        throw NotImplementedException("Writer type is not implemented\n");
    }
    ChunkWriter *chunk_writer = new ChunkWriter(*ims_writer);

    ChunkStager *chunk_stager = new MDStager(*chunk_reader, *chunk_writer);
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
    delete ims_writer;
    delete ims_reader;
    delete py_runner;

    MPI_Finalize();
    return 0;
}

