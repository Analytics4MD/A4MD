#ifndef __COMMON_H__
#define __COMMON_H__
#include <vector>
#include <string>
#include "exceptions.h"
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/vector.hpp>

//typedef struct {
//    const char *data;
//    int size = 0;
//    int chunk_id=0;
//} Chunk;
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

#endif
