#ifndef __PLUMED_CHUNKER_H__
#define __PLUMED_CHUNKER_H__
#include <vector>
#include <list>
#include "common.h"
#include "chunker.h"
#include <boost/serialization/list.hpp>

class PlumedChunker : public Chunker
{
    private:
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const PlumedChunker &pc);
        //std::vector<Chunk> m_chunks;
        std::list<Chunk*> m_chunks;
        ChunkArray m_chunk_array;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) override
        {
            ar.register_type(static_cast<PLMDChunk *>(NULL));
            ar & m_chunks;
        }
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
        void print() override
        {
            for (auto i:m_chunks)
                i->print();
        }
};

#endif
