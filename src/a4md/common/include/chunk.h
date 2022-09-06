#ifndef __CHUNK_H__
#define __CHUNK_H__
#include <vector>
#include <string>
#include <iostream>
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
        ar & m_id;
    }

    protected:
        unsigned long int m_id;

    public:
        Chunk(){}
        Chunk(unsigned long int id) :
             m_id(id)
        {
        }
        virtual ~Chunk()
        {
        }

        virtual void print()
        {
            printf("chunk id %lu\n",  m_id);
        }

        int get_chunk_id()
        {
            return m_id;
        }

        void set_chunk_id(unsigned long int id)
        {
            m_id = id;
        }

        virtual void append(Chunk* other_chunk)
        {
        }
};

#endif
