#include "voronoi_analyzer.h"


VoronoiAnalyzer::VoronoiAnalyzer(ChunkReader & chunk_reader,
                                 PyVoronoiAnalyzer & py_analyzer)
: ChunkAnalyzer(chunk_reader),
  m_py_analyzer(py_analyzer)
{
}

void VoronoiAnalyzer::analyze(Chunk* chunk)
{
    PLMDChunk *plmdchunk = dynamic_cast<PLMDChunk *>(chunk);
    //printf("Printing typecasted chunk\n");
    //chunk->print();
    auto x_positions = plmdchunk->get_x_positions();
    auto y_positions = plmdchunk->get_y_positions();
    auto z_positions = plmdchunk->get_z_positions();
    auto types_vector = plmdchunk->get_types();
    int* types = types_vector.data();

    //for (int i=0;i< types_vector.size(); i++)
    //    printf("type: %i ",types[i]);
    //printf("\n----=======Positions\n");
    //for (auto pos:positions)
    //    printf("%f %f %f \n",std::get<0>(pos), std::get<1>(pos),std::get<2>(pos));
    //printf("----=======Positions end\n");
    double lx, ly, lz, xy, xz, yz; //xy, xz, yz are tilt factors 
    lx = plmdchunk->get_box_lx();
    ly = plmdchunk->get_box_ly();
    lz = plmdchunk->get_box_lz();
    xy = plmdchunk->get_box_xy(); // 0 for orthorhombic
    xz = plmdchunk->get_box_xz(); // 0 for orthorhombic
    yz = plmdchunk->get_box_yz(); // 0 for orthorhombic
    int step = plmdchunk->get_timestep();
    
    m_py_analyzer.analyze_frame(types,
                                x_positions,
                                y_positions,
                                z_positions,
                                lx,
                                ly,
                                lz,
                                xy,
                                xz,
                                yz,
                                step);

}
