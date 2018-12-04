#ifndef __COMMON_H__
#define __COMMON_H__
#include <vector>
#include <string>
#include "exceptions.h"
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>


class Chunk
{
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & m_step;
    }
    protected:
        int m_step;
        Chunk(){}
        Chunk(const int step) :
             m_step(step)
        {
        }
    public:
        virtual void print()
        {
            std::cout << "step " << m_step << std::endl;
        }

        int get_chunk_id()
        {
            return m_step;
        }
};

class PLMDChunk : public Chunk
{
    private:
        friend class boost::serialization::access;
        // When the class Archive corresponds to an output archive, the
        // & operator is defined similar to <<.  Likewise, when the class Archive
        // is a type of input archive the & operator is defined similar to >>.
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<Chunk>(*this); 
            ar & m_types;
        }
        std::vector<int> m_types;
        std::vector<double> m_x_cords;
        std::vector<double> m_y_cords;
        std::vector<double> m_z_cords;
    public:
        PLMDChunk() : Chunk(){}
        PLMDChunk(const int step,
                  const std::vector<int> & m_types) :
                  Chunk(step), m_types(m_types)
        {
        } 
        void print()
        {
            Chunk::print();
            std::cout << "types : ";
            for (auto i: m_types)
                std::cout << i << ' ';
            std::cout << std::endl;
        }
};

class ChunkArray
{
    private:
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const ChunkArray &ca);
        std::list<Chunk*> m_chunks;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) override
        {
            ar.register_type(static_cast<PLMDChunk *>(NULL));
            ar & m_chunks;
        }
    public:
        ChunkArray(){}
        void print()
        {
            for (auto i:m_chunks)
                i->print();
        }
        void append(Chunk* chunk)
        {
            m_chunks.insert(m_chunks.end(), chunk);
        }

        int get_chunk_id()
        {
            if (m_chunks.size() > 0)
                return m_chunks.back()->get_chunk_id();
            else
                return 0;
        }

};

#endif
