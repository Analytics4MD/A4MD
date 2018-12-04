#ifndef __PLUMED_CHUNKER_H__
#define __PLUMED_CHUNKER_H__
#include <vector>
#include "chunker.h"

class PlumedChunker : public Chunker
{
    private:
        ChunkArray m_chunk_array;
    public:
        PlumedChunker();
        ~PlumedChunker();
        void initialize() override;
        void finalize() override;
        std::vector<Chunk> chunks_from_file(int num_chunks=1) override;
        ChunkArray get_chunk_array(int num_chunks=1);
        void append(int step,
                    std::vector<int> types,
                    std::vector<double> x_cords,
                    std::vector<double> y_cords,
                    std::vector<double> z_cords);
};

#endif
