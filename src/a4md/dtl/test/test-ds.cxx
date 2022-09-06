#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include "../include/dataspaces_writer.h"
#include "../include/dataspaces_reader.h"
#include "../../common/include/md_chunk.h"
#include "../../common/include/cv_chunk.h"
#include <vector>
#include <spawn.h>
#include <sys/wait.h>
#include <fstream>
#include <errno.h>
#define errExit(msg)    do { perror(msg); \
                             exit(EXIT_FAILURE); } while (0)
#define errExitEN(en, msg) \
                        do { errno = en; perror(msg); \
                             exit(EXIT_FAILURE); } while (0)

using namespace std;

void ds_write_and_read()
{
    MPI_Init(NULL,NULL);
    ChunkWriter* chunk_writer;
    ChunkReader* chunk_reader;
    unsigned long int total_chunks = 2;
    chunk_writer = new DataSpacesWriter(1, 1, total_chunks, MPI_COMM_WORLD);
    chunk_reader = new DataSpacesReader(2, 1, total_chunks, MPI_COMM_WORLD);

    unsigned long int current_chunk_id = 0;
    int step = 1;
    std::vector<int> types = { 2, 1, 1 };
    std::vector<double> x_positions = { 0.1, 0.1, 0.1 };
    std::vector<double> y_positions = { 0.2, 0.2, 0.2 };
    std::vector<double> z_positions = { 0.2, 0.2, 0.2 };
    double lx, ly, lz, xy, xz, yz;
    lx=ly=lz=xy=xz=yz=1.0;
    lx=1.5;

    Chunk* chunk = new MDChunk(current_chunk_id,step, types, x_positions, y_positions, z_positions, lx, ly, lz, xy, xz, yz);
    MDChunk *md_chunk = dynamic_cast<MDChunk *>(chunk);
    
    std::vector<Chunk*> chunks = {chunk};
    chunk_writer->write_chunks(chunks);
    std::vector<Chunk*> recv_chunks = chunk_reader->read_chunks(current_chunk_id, current_chunk_id);

    MDChunk *recv_md_chunk = dynamic_cast<MDChunk *>(recv_chunks.front());
    

    REQUIRE( md_chunk->get_chunk_id() == recv_md_chunk->get_chunk_id() );
    REQUIRE( md_chunk->get_timestep() == recv_md_chunk->get_timestep() );
    REQUIRE( md_chunk->get_types()[0] == recv_md_chunk->get_types()[0] );
    REQUIRE( md_chunk->get_types().size() == recv_md_chunk->get_types().size() );
    REQUIRE( md_chunk->get_x_positions()[0] == recv_md_chunk->get_x_positions()[0] );
    REQUIRE( md_chunk->get_x_positions().size() == recv_md_chunk->get_x_positions().size() );
    REQUIRE( md_chunk->get_box_lx() == recv_md_chunk->get_box_lx() );
    delete chunk;

    current_chunk_id = 1;
    std::vector<double> cv_values = { 0.1, 0.1, 0.1, 0.2, 0.2, 0.2 };
    chunk = new CVChunk(current_chunk_id, cv_values);
    CVChunk *cv_chunk = dynamic_cast<CVChunk*>(chunk);
    chunks.clear();
    recv_chunks.clear();

    chunks.push_back(chunk);
    chunk_writer->write_chunks(chunks);
    recv_chunks = chunk_reader->read_chunks(current_chunk_id, current_chunk_id);
    CVChunk *recv_cv_chunk = dynamic_cast<CVChunk *>(recv_chunks.front());
    // recv_cv_chunk->print();

    REQUIRE( cv_chunk->get_chunk_id() == recv_cv_chunk->get_chunk_id() );
    REQUIRE( cv_chunk->get_cv_values()[0] == recv_cv_chunk->get_cv_values()[0] );
    REQUIRE( cv_chunk->get_cv_values().size() == recv_cv_chunk->get_cv_values().size() );
    delete chunk;
    
    delete chunk_writer;
    delete chunk_reader;

    MPI_Finalize(); 
    printf("Completed dataspaces write and read successfully\n");
}

TEST_CASE( "DS Write-Read Test", "[dtl]" ) 
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
      file_ << "max_versions = 1\n";
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
    if (status == 0) {
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
