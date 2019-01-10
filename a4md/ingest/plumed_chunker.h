#ifndef __PLUMED_CHUNKER_H__
#define __PLUMED_CHUNKER_H__
#include <vector>
#include <queue>
#include "chunker.h"

class PlumedChunker : public Chunker
{
    private:
        std::queue<Chunk*> m_chunkq;
    public:
        std::vector<Chunk*> get_chunks(int num_chunks) override;
        //ChunkArray get_chunk_array(int num_chunks=1);
        void append(unsigned long int chunk_id,
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
                    double box_yz);
};

#endif
