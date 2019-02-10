#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include "chunk.h"
#include <vector>

TEST_CASE( "Chunk Tests", "[common]" ) 
{
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

    REQUIRE( chunk->get_chunk_id() == 0 );
    REQUIRE( md_chunk.get_timestep() == 1 );
    REQUIRE( md_chunk.get_types()[0] == 2 );
    REQUIRE( md_chunk.get_x_positions()[0] == 0.1 );
    REQUIRE( md_chunk.get_box_lx() == 1.5 );
}
