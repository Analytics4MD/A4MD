#ifndef __SERIALIZABLE_CHUNK_H__
#define __SERIALIZABLE_CHUNK_H__
#include <vector>
#include <string>
#include <iostream>
#include "md_chunk.h"
#include "cv_chunk.h"
#include <boost/serialization/shared_ptr.hpp>

class SerializableChunk
{
    private:
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const SerializableChunk &ca);
        std::shared_ptr<Chunk> m_chunk;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            // IMPORTANT: Any chunk subclass that needs to be serialized has to have an entry here.
            ar.register_type(static_cast<MDChunk *>(NULL));
            ar.register_type(static_cast<CVChunk *>(NULL));
            ar & m_chunk;
        }

    public:
        SerializableChunk()
        {
        }

        SerializableChunk(std::shared_ptr<Chunk> chunk)
        :m_chunk(chunk)
        {
        }

        ~SerializableChunk()
        {
            printf("(%s): destructor\n", __func__);
        }

        void print()
        {
            m_chunk->print();
        }

        unsigned long int get_chunk_id()
        {
            m_chunk->get_chunk_id();
        }

        std::shared_ptr<Chunk> get_chunk()
        {
            return m_chunk;
        }

};

#endif
