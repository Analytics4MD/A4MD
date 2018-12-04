//#include "ingest.h"
#include "mpi.h"
#include "dataspaces_writer.h"
#include "plumed_chunker.h"

int main (int argc, const char** argv)
{
  MPI_Init(NULL,NULL);
  //Ingest ingest;
  //ingest.run();
  char* var_name = "test_var";
  DataSpacesWriter dataspaces_writer_ptr = DataSpacesWriter(var_name);

  std::vector<double> x_positions = {0.1,1.0,2.0}; 
  std::vector<double> y_positions = {10.0,20.0,30.0}; 
  std::vector<double> z_positions = {111.0,123.0,454.0}; 
 
  std::vector<int> types = {0,0,0}; 
  int step = 100; 
  PlumedChunker chunker = PlumedChunker();
  chunker.append(step, 
                 types, 
                 x_positions, 
                 y_positions, 
                 z_positions); 

  step = 101;
  x_positions[0] = 0.2;
  types[1] = 1;
  chunker.append(step, 
                 types, 
                 x_positions, 
                 y_positions, 
                 z_positions);
  //std::vector<Chunk> chunks = chunker.chunks_from_file();
  //printf("chunks---- length %i\n",chunks.size());
  auto chunk_array = chunker.get_chunk_array();
  dataspaces_writer_ptr.write_chunks(chunk_array);

  //dataspaces_writer_ptr.write_chunks(chunker);
 
  printf("Initialized dataspace writer\n");
  MPI_Finalize();
  return 0;
}
