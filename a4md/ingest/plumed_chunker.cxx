#include "plumed_chunker.h"


void PlumedChunker::append(unsigned long int id,
                           int time_step,
                           std::vector<int> types,
                           std::vector<double> x_cords,
                           std::vector<double> y_cords,
                           std::vector<double> z_cords,
                           double box_lx,
                           double box_ly,
                           double box_lz,
                           double box_xy,
                           double box_xz,
                           double box_yz)


{
    Chunk* chunk = new MDChunk(id,
                                 time_step,
                                 types,
                                 x_cords,
                                 y_cords,
                                 z_cords,
                                 box_lx,
                                 box_ly,
                                 box_lz,
                                 box_xy,
                                 box_xz,
                                 box_yz);
    m_chunkq.push(chunk);
}
