#include "dataspaces_writer.h"
#include "dataspaces_reader.h"
#include <catch2/catch.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <fstream>
#include <errno.h>
#include "md_runner.h"
#include "pdb_chunker.h"


#define errExit(msg)    do { perror(msg); \
                             exit(EXIT_FAILURE); } while (0)
#define errExitEN(en, msg) \
                        do { errno = en; perror(msg); \
                             exit(EXIT_FAILURE); } while (0)

using namespace std;
void ds_write_and_read()
{
    MPI_Init(NULL,NULL);
    DataSpacesWriter* dataspaces_writer_ptr;
    DataSpacesReader* dataspaces_reader_ptr;
    unsigned long int current_chunk_id = 1;

    std::string name = "ingest";
    std::string func = "extract_frame";
    std::string py_path = "./a4md/ingest/test";
    std::string file_path = "./a4md/ingest/test/test.pdb";

    // PyRunner* py_runner = new MDRunner((char*)name.c_str(), 
    //                                    (char*)func.c_str(),
    //                                    (char*)py_path.c_str());
    PDBChunker* pdb_chunker = new PDBChunker((char*)name.c_str(), (char*)func.c_str(), (char*)py_path.c_str(), (char*)file_path.c_str(), 0, 0);
    //unsigned long int id = 0;
    // int result = pdb_chunker->extract_chunk();
    std::vector<std::shared_ptr<Chunk>> chunk_vector = pdb_chunker->read_chunks(1, 1);
    std::shared_ptr<Chunk> chunk = chunk_vector.front(); 
    std::shared_ptr<MDChunk> md_chunk = std::dynamic_pointer_cast<MDChunk>(chunk);
    //md_chunk->print();
    
    unsigned long int total_chunks = 1;
    dataspaces_writer_ptr = new DataSpacesWriter(1, 1, total_chunks, MPI_COMM_WORLD);
    dataspaces_reader_ptr = new DataSpacesReader(2, 1, total_chunks, MPI_COMM_WORLD);
    std::vector<std::shared_ptr<Chunk>> chunks = {chunk};
    dataspaces_writer_ptr->write_chunks(chunks);
    
    std::vector<std::shared_ptr<Chunk>> recieved_chunks = dataspaces_reader_ptr->read_chunks(current_chunk_id, current_chunk_id);
    
    for (auto chunk: recieved_chunks)
    {
        std::shared_ptr<MDChunk> recieved_chunk = std::dynamic_pointer_cast<MDChunk>(chunk);
        //recieved_chunk->print();
        auto recieved_x_positions = recieved_chunk->get_x_positions();
        auto recieved_y_positions = recieved_chunk->get_y_positions();
        auto recieved_z_positions = recieved_chunk->get_z_positions();
        auto recieved_types_vector = recieved_chunk->get_types();
        
        double recieved_lx = recieved_chunk->get_box_lx();
        double recieved_ly = recieved_chunk->get_box_ly();
        double recieved_lz = recieved_chunk->get_box_lz();
        double recieved_hx = recieved_chunk->get_box_hx(); // 0 for orthorhombic
        double recieved_hy = recieved_chunk->get_box_hy(); // 0 for orthorhombic
        double recieved_hz = recieved_chunk->get_box_hz(); // 0 for orthorhombic
        int recieved_step = recieved_chunk->get_timestep();
        
        REQUIRE( chunk->get_chunk_id() == recieved_chunk->get_chunk_id() );
        REQUIRE( md_chunk->get_timestep() == recieved_chunk->get_timestep() );
        REQUIRE( md_chunk->get_types()[0] == recieved_chunk->get_types()[0] );
        REQUIRE( md_chunk->get_x_positions()[0] == recieved_chunk->get_x_positions()[0] );
        REQUIRE( md_chunk->get_box_lx() == recieved_chunk->get_box_lx() );
    }
    MPI_Finalize(); 
    printf("Completed dataspaces write and read successfully\n");
}

TEST_CASE( "DS PDBChunker Ingest Test", "[ingest]" ) 
{
    pid_t child_pid;
    extern char **environ;
    char *cmd = (char*)"dataspaces_server -s 1 -c 1";
    char *argv[] = {(char*)"sh", (char*)"-c", cmd, NULL};
    int status;

    ofstream file_("dataspaces.conf");
    if (file_.is_open())
    {
        file_ << " \n";
        file_ << "ndim = 1\n";
        file_ << "dims = 100000\n";
        file_ << "max_versions = 10\n";
        file_ << "max_readers = 1\n";
        file_ << "lock_type = 1\n";
        file_ << "hash_version = 1\n";
    } 
    file_.close();
    
    printf("Run command: %s\n", cmd);
    status = posix_spawnp(&child_pid, "/bin/sh", NULL, NULL, argv, environ);
    // --------========= NOT SURE WHY THIS MY CHILD PID is +1 ===========--------------
    // TODO: fix the +1 issue
    child_pid += 1;
    printf("Started dataspaces_server with  pid: %i\n", child_pid);

    if (status == 0) 
    {
        printf("sleeping 3 seconds\n");
        sleep(3);
        printf("done sleeping\n");
        ds_write_and_read();

        status = kill(child_pid, SIGTERM);
        printf("Child exited with status %i\n", status);
    }
    else 
    {
        printf("posix_spawn: %s\n", strerror(status));
    }
    
}
