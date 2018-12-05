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
        PLMDChunk() : Chunk(){}
        PLMDChunk(const int step,
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
                  Chunk(step),
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
            printf("--------==========PLMDChunk::print start=============--------------\n");
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
            printf("--------==========PLMDChunk::print end=============--------------\n");
        }
        
        std::vector<int> get_types()
        {
            return m_types;
        }

        std::vector<std::tuple<double, double, double> > get_positions()
        {
            std::vector<std::tuple<double, double, double> > positions;
            //printf("----=====Inside get positions size = %i\n",m_x_cords.size());
            for( auto iterator = m_x_cords.begin() ; iterator != m_x_cords.end() ; ++iterator )
            {
                auto position = std::distance( m_x_cords.begin(), iterator ) ;
                auto pos_tuple = std::make_tuple(m_x_cords[position],m_y_cords[position],m_z_cords[position]);
                positions.push_back(pos_tuple);
            }
            return positions;
        }

        double get_box_lx(){ return m_box_lx; }
        double get_box_ly(){ return m_box_ly; }
        double get_box_lz(){ return m_box_lz; }
        double get_box_xy(){ return m_box_xy; }
        double get_box_xz(){ return m_box_xz; }
        double get_box_yz(){ return m_box_yz; }
};

class ChunkArray
{
    private:
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const ChunkArray &ca);
        std::list<Chunk*> m_chunks;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
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

        std::list<Chunk*> get_chunks()
        {
            return m_chunks;
        }

};

#endif
