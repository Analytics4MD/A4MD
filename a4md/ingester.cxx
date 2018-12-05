//#include "ingest.h"
#include "mpi.h"
#include "dataspaces_writer.h"
#include "plumed_chunker.h"

int main (int argc, const char** argv)
{
  MPI_Init(NULL,NULL);
  char* var_name = "test_var";
  DataSpacesWriter dataspaces_writer_ptr = DataSpacesWriter(var_name);

  std::vector<double> x_positions = {0.1,1.0,2.0,3.0}; 
  std::vector<double> y_positions = {10.0,20.0,30.0,40.0}; 
  std::vector<double> z_positions = {111.0,123.0,454.0,645.0}; 
 
  double lx,ly,lz,xy,xz,yz;
  lx=ly=lz=10.0;
  xy=xz=yz=1.0;
  std::vector<int> types = {0,0,0}; 
  int step = 100; 
  PlumedChunker chunker = PlumedChunker();
  chunker.append(step, 
                 types, 
                 x_positions, 
                 y_positions, 
                 z_positions,
                 lx,ly,lz,
                 xy,xz,yz); 

  auto chunk_array = chunker.get_chunk_array();
  dataspaces_writer_ptr.write_chunks(chunk_array);
  printf("Write 1 done\n");
 
  step = 101;
  x_positions[0] = 0.2;
  types[1] = 1;
  chunker.append(step, 
                 types, 
                 x_positions, 
                 y_positions, 
                 z_positions,
                 lx,ly,lz,
                 xy,xz,yz);
  chunk_array = chunker.get_chunk_array();
  dataspaces_writer_ptr.write_chunks(chunk_array);
  printf("Write 2 done\n");

  MPI_Finalize();
  return 0;
}
