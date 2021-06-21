#ifndef __MD_CHUNK_H__
#define __MD_CHUNK_H__
#include <vector>
#include <string>
#include <iostream>
#include "chunk.h"

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
            ar & m_box_hx;
            ar & m_box_hy;
            ar & m_box_hz;
        }
        int m_timestep;
        std::vector<int> m_types;
        std::vector<double> m_x_cords;
        std::vector<double> m_y_cords;
        std::vector<double> m_z_cords;
        double m_box_lx;
        double m_box_ly;
        double m_box_lz;
        double m_box_hx;
        double m_box_hy;
        double m_box_hz;
    public:
        MDChunk() : Chunk(){}
        MDChunk(unsigned long int id,
                int timestep,
                std::vector<int> & types,
                std::vector<double> & x_cords,
                std::vector<double> & y_cords,
                std::vector<double> & z_cords,
                double box_lx,
                double box_ly,
                double box_lz,
                double box_hx,
                double box_hy,
                double box_hz) :
                Chunk(id),
                m_timestep(timestep),
                m_types(types),
                m_x_cords(x_cords),
                m_y_cords(y_cords),
                m_z_cords(z_cords),
                m_box_lx(box_lx),
                m_box_ly(box_ly),
                m_box_lz(box_lz),
                m_box_hx(box_hx),
                m_box_hy(box_hy),
                m_box_hz(box_hz)
        {
        } 
        ~MDChunk()
        {
            printf("(%s): destructor\n", __func__);
        }

        void print()
        {
            printf("--------==========MDChunk::print start=============--------------\n");
            Chunk::print();
            printf("(%s): types : ", __func__);
            for (auto i: m_types)
                printf("%d ", i);
            printf("\n");
            printf("(%s): positions :\n", __func__);
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
        double get_box_hx(){ return m_box_hx; }
        double get_box_hy(){ return m_box_hy; }
        double get_box_hz(){ return m_box_hz; }
        double get_timestep(){ return m_timestep; }
};

#endif
