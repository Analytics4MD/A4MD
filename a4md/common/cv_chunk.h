#ifndef __CV_CHUNK_H__
#define __CV_CHUNK_H__
#include <vector>
#include <string>
#include <iostream>
#include "chunk.h"

class CVChunk : public Chunk
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
            ar & m_cv_vals;
        }
        std::vector<double> m_cv_vals;

    public:
        CVChunk() : Chunk(){}
        CVChunk(unsigned long int id,
                std::vector<double> & cv_vals) :
                Chunk(id),
                m_cv_vals(cv_vals)
        {
        } 
        CVChunk(unsigned long int id) : Chunk(id) 
        {
        }
        ~CVChunk()
        {
            printf("---===== Called destructor of CVChunk\n");
        }

        void print()
        {
            printf("---===== CVChunk::print start\n");
            Chunk::print();
            for( auto cv_val : m_cv_vals)
            {
                printf("%f ", cv_val);
            }
            printf("\n");
            printf("---===== CVChunk::print end\n");
        }

        void append(double cv_val)
        {
            m_cv_vals.push_back(cv_val);
        }

        void append(Chunk* other_chunk)
        {
            CVChunk *other_cv_chunk = dynamic_cast<CVChunk*>(other_chunk);
            std::vector<double> other_cv_vals = other_cv_chunk->get_cv_values();
            m_cv_vals.insert(m_cv_vals.end(), other_cv_vals.begin(), other_cv_vals.end());
        }

        std::vector<double> get_cv_values(){ return m_cv_vals; }
};

#endif /* __CV_CHUNK_H__ */
