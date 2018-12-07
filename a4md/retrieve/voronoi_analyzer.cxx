#include "voronoi_analyzer.h"


VoronoiAnalyzer::VoronoiAnalyzer(ChunkReader & chunk_reader,
                                 std::string module_name,
                                 std::string function_name)
: ChunkAnalyzer(chunk_reader),
  m_module_name(module_name),
  m_function_name(function_name)
{
}

void VoronoiAnalyzer::analyze(Chunk* chunk)
{
    PLMDChunk *plmdchunk = dynamic_cast<PLMDChunk *>(chunk);
    //printf("Printing typecasted chunk\n");
    //chunk->print();
    POS_VEC positions = plmdchunk->get_positions();
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
    
    m_py_analyzer.analyze_frame((char*)m_module_name.c_str(),
                                (char*)m_function_name.c_str(),
                                types,
                                positions,
                                lx,
                                ly,
                                lz,
                                xy,
                                xz,
                                yz,
                                step);

}
