//#include "ingest.h"
#include "mpi.h"
#include "dataspaces_writer.h"
#include "plumed_chunker.h"

int main (int argc, const char** argv)
{
  MPI_Init(NULL,NULL);
  std::string var_name = "test_var";
  if (argc != 2)
    printf("ERROR: Expecting number of simulation steps as command line argument\n");
  unsigned long int n_chunks = atoi(argv[1]);
  DataSpacesWriter dataspaces_writer_ptr = DataSpacesWriter(1,(char*)var_name.c_str(), n_chunks, MPI_COMM_WORLD);

   for (int chunkid=0;chunkid<n_chunks;chunkid++)
  {
      std::vector<double> x_positions = {1.0*chunkid,1.1*chunkid,1.2*chunkid,1.3*chunkid}; 
      std::vector<double> y_positions = {0.0*chunkid,0.1*chunkid,0.2*chunkid,0.3*chunkid}; 
      std::vector<double> z_positions = {10.0*chunkid,10.1*chunkid,10.2*chunkid,10.3*chunkid}; 

      double lx,ly,lz,xy,xz,yz;
      lx=ly=lz=10.0;
      xy=xz=yz=1.0;
      std::vector<int> types = {0,0,0}; 
      int step = chunkid; 
      PlumedChunker chunker = PlumedChunker();
      chunker.append(chunkid,
                     step, 
                     types, 
                     x_positions, 
                     y_positions, 
                     z_positions,
                     lx,ly,lz,
                     xy,xz,yz); 

      auto chunk_array = chunker.get_chunks(1);//_array();
      dataspaces_writer_ptr.write_chunks(chunk_array);
      printf("Write %i done\n",chunkid);
  } 
  
  MPI_Finalize();
  return 0;
}
