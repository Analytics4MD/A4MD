#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include "dataspaces_writer.h"
#include "dataspaces_reader.h"
#include "md_chunk.h"
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
    DataSpacesWriter* dataspaces_writer_ptr;
    DataSpacesReader* dataspaces_reader_ptr;
    unsigned long int current_chunk_id = 0;
    int step = 1;
    std::vector<int> types = { 2, 1, 1 };
    std::vector<double> x_positions = { 0.1, 0.1, 0.1 };
    std::vector<double> y_positions = { 0.2, 0.2, 0.2 };
    std::vector<double> z_positions = { 0.2, 0.2, 0.2 };
    double lx, ly, lz, xy, xz, yz;
    lx=ly=lz=xy=xz=yz=1.0;
    lx=1.5;

    MDChunk md_chunk(current_chunk_id,
                           step,
                           types,
                           x_positions,
                           y_positions,
                           z_positions,
                           lx,
                           ly,
                           lz,
                           xy,
                           xz,
                           yz);
    Chunk* chunk = &md_chunk; 

    unsigned long int total_chunks = 1;
    dataspaces_writer_ptr = new DataSpacesWriter(1, 1, total_chunks, MPI_COMM_WORLD);
    dataspaces_reader_ptr = new DataSpacesReader(2, 1, total_chunks, MPI_COMM_WORLD);
    std::vector<Chunk*> chunks = {chunk};
    dataspaces_writer_ptr->write_chunks(chunks);
    std::vector<Chunk*> recieved_chunks = dataspaces_reader_ptr->read_chunks(current_chunk_id, current_chunk_id);
    for (Chunk* chunk: recieved_chunks)
    {
      MDChunk *recieved_chunk = dynamic_cast<MDChunk *>(chunk);
      //printf("Printing typecasted chunk\n");
      //chunk->print();
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
      REQUIRE( md_chunk.get_timestep() == recieved_chunk->get_timestep() );
      REQUIRE( md_chunk.get_types()[0] == recieved_chunk->get_types()[0] );
      REQUIRE( md_chunk.get_x_positions()[0] == recieved_chunk->get_x_positions()[0] );
      REQUIRE( md_chunk.get_box_lx() == recieved_chunk->get_box_lx() );
    }
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
