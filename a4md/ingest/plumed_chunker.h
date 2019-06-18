#ifndef __PLUMED_CHUNKER_H__
#define __PLUMED_CHUNKER_H__
#include <vector>
#include "chunker.h"

class PlumedChunker : public Chunker
{
    public:
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

        std::vector<Chunk*> get_chunks(int num_chunks);

        std::vector<Chunk*> get_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to) override;
};

#endif
