#ifndef __COMMON_H__
#define __COMMON_H__
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
        Chunk(const unsigned long int id) :
             m_id(id)
        {
        }

        virtual void print()
        {
            std::cout << "chunk id " << m_id << std::endl;
        }

        int get_chunk_id()
        {
            return m_id;
        }
};

class MDChunk : public Chunk
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
            ar & m_timestep;
            ar & m_types;
            ar & m_x_cords;
            ar & m_y_cords;
            ar & m_z_cords;
            ar & m_box_lx;
            ar & m_box_ly;
            ar & m_box_lz;
            ar & m_box_xy;
            ar & m_box_xz;
            ar & m_box_yz;
        }
        int m_timestep;
        std::vector<int> m_types;
        std::vector<double> m_x_cords;
        std::vector<double> m_y_cords;
        std::vector<double> m_z_cords;
        double m_box_lx;
        double m_box_ly;
        double m_box_lz;
        double m_box_xy;
        double m_box_xz;
        double m_box_yz;
    public:
        MDChunk() : Chunk(){}
        MDChunk(const unsigned long int id,
                  const int timestep,
                  const std::vector<int> & types,
                  const std::vector<double> & x_cords,
                  const std::vector<double> & y_cords,
                  const std::vector<double> & z_cords,
                  double box_lx,
                  double box_ly,
                  double box_lz,
                  double box_xy,
                  double box_xz,
                  double box_yz) :
                  Chunk(id),
                  m_timestep(timestep),
                  m_types(types),
                  m_x_cords(x_cords),
                  m_y_cords(y_cords),
                  m_z_cords(z_cords),
                  m_box_lx(box_lx),
                  m_box_ly(box_ly),
                  m_box_lz(box_lz),
                  m_box_xy(box_xy),
                  m_box_xz(box_xz),
                  m_box_yz(box_yz)
        {
        } 

        void print()
        {
            printf("--------==========MDChunk::print start=============--------------\n");
            Chunk::print();
            std::cout << "types : ";
            for (auto i: m_types)
                std::cout << i << ' ';
            std::cout << std::endl;
            std::cout << "positions :\n";
            for( auto iterator = m_x_cords.begin() ; iterator != m_x_cords.end() ; ++iterator )
            {
                int position = std::distance( m_x_cords.begin(), iterator ) ;
                printf("[%i] %f %f %f \n",position, m_x_cords[position],m_y_cords[position],m_z_cords[position]);
            }
            printf("--------==========MDChunk::print end=============--------------\n");
        }
        
        std::vector<int> get_types()
        {
            return m_types;
        }

        std::vector<double> get_x_positions(){ return m_x_cords; }
        std::vector<double> get_y_positions(){ return m_y_cords; }
        std::vector<double> get_z_positions(){ return m_z_cords; }

        double get_box_lx(){ return m_box_lx; }
        double get_box_ly(){ return m_box_ly; }
        double get_box_lz(){ return m_box_lz; }
        double get_box_xy(){ return m_box_xy; }
        double get_box_xz(){ return m_box_xz; }
        double get_box_yz(){ return m_box_yz; }
        double get_timestep(){ return m_timestep; }
};

class SerializableChunk
{
    private:
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const SerializableChunk &ca);
        Chunk* m_chunk;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            // IMPORTANT: Any chunk subclass that needs to be serialized has to have an entry here.
            ar.register_type(static_cast<MDChunk *>(NULL));
            ar & m_chunk;
        }

    public:
        SerializableChunk()
        {
        }

        SerializableChunk(Chunk* chunk)
        :m_chunk(chunk)
        {
        }

        void print()
        {
            m_chunk->print();
        }

        unsigned long int get_chunk_id()
        {
            m_chunk->get_chunk_id();
        }

        Chunk* get_chunk()
        {
            return m_chunk;
        }

};


#endif
