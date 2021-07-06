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
    private:
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
        Chunk(unsigned long int id) : m_id(id) { }
        virtual ~Chunk() 
        {
            printf("(%s): destructor\n", __func__);
        }

        virtual void print()
        {
            printf("(%s): chunk id %lu\n", __func__, m_id);
        }

        int get_chunk_id()
        {
            return m_id;
        }

        void set_chunk_id(unsigned long int id)
        {
            m_id = id;
        }

        virtual void append(std::shared_ptr<Chunk> other_chunk) { }
};

void print(Chunk* chunk);

class TestChunk : public Chunk
{
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<Chunk>(*this);
            ar & m_timestep;
        }
        int m_timestep;
    public:
        TestChunk() {}
        TestChunk(unsigned long int id, int timestep) : Chunk(id), m_timestep(timestep) {}
        ~TestChunk() 
        { 
            printf("(%s): destructor\n", __func__);
        }
        void print()
        {
            Chunk::print();
            printf("(%s): time step %d\n", __func__, m_timestep);
        }
};


#endif
