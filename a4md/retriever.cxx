#include "retrieve.h"
#include "mpi.h"
#include "dataspaces_reader.h"
#include "voronoi_analyzer.h"
#include "md_retriever.h"
#include <unistd.h>


ChunkAnalyzer* analyzer_factory(int argc, const char** argv)
{
    

    std::string analyzer_name = "voronoi_analyzer";
    std::string reader_type = "dataspaces";
    std::string var_name = "test_var";

    ChunkAnalyzer* chunk_analyzer;
    ChunkReader* chunk_reader;
 
    int n_steps = atoi(argv[3]);
    int n_stride = atoi(argv[4]);
    int n_analysis_stride = 1;
    unsigned long int total_chunks = n_steps/n_stride/n_analysis_stride;
    if (reader_type == "dataspaces")
    {
        printf("---======== Initializing dataspaces reader\n");
        Chunker * chunker = new DataSpacesReader((char*)var_name.c_str(), total_chunks);
        printf("---======== Initialized dataspaces reader\n");
        chunk_reader = new ChunkReader(* chunker);
        printf("---======== Initialized chunkreader\n");
    }
    else
    {
        throw NotImplementedException("Reader type is not implemented");
    }

    if(analyzer_name == "voronoi_analyzer")
    {
        std::string name((char*)argv[1]);
        std::string func((char*)argv[2]);
        chunk_analyzer = new VoronoiAnalyzer(* chunk_reader, name, func);
        printf("---======== Initialized voronoi analyzer\n");
    }
    else
    {
        throw NotImplementedException("Analyzer type is not implemented");
    }

    return chunk_analyzer;
}

Retriever* retriever_factory (int argc, const char** argv)
{
    ChunkAnalyzer * analyzer = analyzer_factory(argc, argv);
    int n_steps = atoi(argv[3]);
    int n_stride = atoi(argv[4]);
    int n_analysis_stride = 1;
    
    printf("Recieved n_steps = %i from user in Retriever\n",n_steps);
    Retriever * retriever = new MDRetriever(* analyzer, n_steps, n_stride, n_analysis_stride);
    return retriever;
}

int main (int argc, const char** argv)
{
    MPI_Init(NULL,NULL);
    printf("---======== In Retriever::main()\n");

    Retriever * retriever = retriever_factory(argc,argv);
    retriever->run();
    
    MPI_Finalize();
    return 0;
}
