#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <string>
#include <vector>
#include "chunk.h"
#include "py_runner.h"
#include "pdb_chunker.h"

TEST_CASE("PyRunner extract_frame Tests", "[ingest]")
{
	std::string name = "load";
	std::string func = "extract_frame";
	// ToDo: unhardcode 
	std::string py_path = "./a4md/ingest/test";
	std::string file_path = "./a4md/ingest/test/test.pdb";

	PyRunner* py_runner = new PyRunner((char*)name.c_str(), 
									   (char*)func.c_str(),
									   (char*)py_path.c_str());
	
	int position = 0;
	Chunk* chunk;
	int result = py_runner->extract_frame((char*)file_path.c_str(), position, chunk);
	printf("New position : %d\n", position);

	MDChunk *plmdchunk = dynamic_cast<MDChunk *>(chunk);
	auto x_positions = plmdchunk->get_x_positions();
	auto y_positions = plmdchunk->get_y_positions();
	auto z_positions = plmdchunk->get_z_positions();
	auto types_vector = plmdchunk->get_types();
	int timestep = plmdchunk->get_timestep();

	REQUIRE( result == 0 );
	REQUIRE( position == 4851 );
	REQUIRE( x_positions.size() == y_positions.size() );
	REQUIRE( y_positions.size() == z_positions.size() );
	REQUIRE( z_positions.size() == types_vector.size() );
	REQUIRE( timestep == 0);
}

TEST_CASE("PDBChunker Tests", "[ingest]")
{
	std::string name = "load";
	std::string func = "extract_frame";
	// ToDo: unhardcode 
	std::string py_path = "./a4md/ingest/test";
	std::string file_path = "./a4md/ingest/test/test.pdb";

	PyRunner* py_runner = new PyRunner((char*)name.c_str(), 
									   (char*)func.c_str(),
									   (char*)py_path.c_str());
	
	PDBChunker* pdb_chunker = new PDBChunker((*py_runner),
											 (char*)file_path.c_str());
	int result = pdb_chunker->extract_chunk();
	std::vector<Chunk*> chunk_vector = pdb_chunker->get_chunks(1);
	Chunk* chunk = chunk_vector.front();
	MDChunk *plmdchunk = dynamic_cast<MDChunk *>(chunk);
	
	auto x_positions = plmdchunk->get_x_positions();
	auto y_positions = plmdchunk->get_y_positions();
	auto z_positions = plmdchunk->get_z_positions();
	auto types_vector = plmdchunk->get_types();
	int timestep = plmdchunk->get_timestep();

	REQUIRE( result == 0 );
	REQUIRE( pdb_chunker->get_position() == 4851 );
	REQUIRE( chunk_vector.size() == 1 );
	REQUIRE( x_positions.size() == y_positions.size() );
	REQUIRE( y_positions.size() == z_positions.size() );
	REQUIRE( z_positions.size() == types_vector.size() );
	REQUIRE( timestep == 0);
}

//int main(int argc, char* argv[])
//TEST_CASE("PDBChunker Tests", "[ingest]")
//{
//	//int catch_session = Catch::Session().run( argc, argv );
//
//	std::string name = "load";
//	std::string func = "extract_frame";
//	//std::string py_path((char*)argv[1]);
//	//std::string file_path((char*)argv[2]);
//	std::string py_path = "./a4md/ingest/test";
//	std::string file_path = "./a4md/ingest/test/test.pdb";
//
//	PyRunner* py_runner = new PyRunner((char*)name.c_str(), 
//									   (char*)func.c_str(),
//									   (char*)py_path.c_str());
//	PDBChunker* pdb_chunker = new PDBChunker((*py_runner),
//											 (char*)file_path.c_str());
//	int result = pdb_chunker->extract_chunk();
//	std::vector<Chunk*> chunk_vector = pdb_chunker->get_chunks(1);
//	Chunk* chunk = chunk_vector.front();
//	MDChunk *plmdchunk = dynamic_cast<MDChunk *>(chunk);
//	
//	auto x_positions = plmdchunk->get_x_positions();
//	auto y_positions = plmdchunk->get_y_positions();
//	auto z_positions = plmdchunk->get_z_positions();
//	auto types_vector = plmdchunk->get_types();
//	int timestep = plmdchunk->get_timestep();
//
//	REQUIRE( result == 0 );
//	REQUIRE( chunk_vector.size() == 1 );
//	REQUIRE( x_positions.size() == 37 );
//	REQUIRE( y_positions.size() == 37 );
//	REQUIRE( z_positions.size() == 37 );
//	REQUIRE( types_vector.size() == 37 );
//	REQUIRE( timestep == 0 );
//	printf("End test\n");
//	//return catch_session;
//}
