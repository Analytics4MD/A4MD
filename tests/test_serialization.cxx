#include <iostream>
#include <sstream>
#include <limits>
#include <random>
#include <vector>
#include <string>
#include <type_traits>
#include "mpi.h"
// #include "chunk.h"
// #include "timer.h"
#include <chrono>
#define timeNow() std::chrono::high_resolution_clock::now()
typedef std::chrono::high_resolution_clock::time_point TimeVar;
typedef std::chrono::duration<double, std::milli> DurationMilli;
// Boost headers
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/export.hpp>
// Cereal headers
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
// Yas headers
/*#include <yas/mem_streams.hpp>
#include <yas/binary_iarchive.hpp>
#include <yas/binary_oarchive.hpp>
#include <yas/std_types.hpp>*/

class BoostChunk
{
    protected:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & m_id;
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
        unsigned long int m_id;
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
        BoostChunk() {}
        BoostChunk(const unsigned long int id,
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
                  m_id(id),
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
        ~BoostChunk()
        {
            printf("---===== Called destructor of BoostChunk\n");
        }

        void print()
        {
            printf("--------==========MDChunk::print start=============--------------\n");
            printf("types : ");
            for (auto i: m_types)
                printf("%d ", i);
            printf("\n");
            printf("positions :\n");
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

//BOOST_CLASS_EXPORT(BoostChunk)

class CerealChunk
{
    protected:
        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & ar, std::uint32_t const version)
        {
            ar(m_id);
            ar(m_timestep);
            ar(m_types);
            ar(m_x_cords);
            ar(m_y_cords);
            ar(m_z_cords);
            ar(m_box_lx);
            ar(m_box_ly);
            ar(m_box_lz);
            ar(m_box_xy);
            ar(m_box_xz);
            ar(m_box_yz);
        }
        unsigned long int m_id;
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
        CerealChunk() {}
        CerealChunk(const unsigned long int id,
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
                  m_id(id),
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
        ~CerealChunk()
        {
            printf("---===== Called destructor of CerealChunk\n");
        }
};

/*class YasChunk
{
    protected:
        template<class Archive>
        void serialize(Archive & ar)
        {
            ar & YAS_OBJECT(m_id);
            ar & YAS_OBJECT(m_timestep);
            ar & YAS_OBJECT(m_types);
            ar & YAS_OBJECT(m_x_cords);
            ar & YAS_OBJECT(m_y_cords);
            ar & YAS_OBJECT(m_z_cords);
            ar & YAS_OBJECT(m_box_lx);
            ar & YAS_OBJECT(m_box_ly);
            ar & YAS_OBJECT(m_box_lz);
            ar & YAS_OBJECT(m_box_xy);
            ar & YAS_OBJECT(m_box_xz);
            ar & YAS_OBJECT(m_box_yz);
        }
        unsigned long int m_id;
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
        YasChunk() {}
        YasChunk(const unsigned long int id,
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
                  m_id(id),
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
        ~YasChunk()
        {
            printf("---===== Called destructor of YasChunk\n");
        }
};*/

int natoms;
BoostChunk *boost_chunk;
CerealChunk *cereal_chunk;
// YasChunk *yas_chunk;


int generate_int()
{
    static std::uniform_int_distribution<int> distribution(std::numeric_limits<int>::min(),std::numeric_limits<int>::max());
    static std::default_random_engine generator;
    return distribution(generator);
}

int generate_double()
{
    static std::uniform_real_distribution<double> distribution(std::numeric_limits<double>::min(),std::numeric_limits<double>::max());
    static std::default_random_engine generator;
    return distribution(generator);
}

std::vector<int> generate_int_vector(size_t size)
{
    std::vector<int> data(size);
    std::generate(data.begin(), data.end(), []() { return generate_int(); });
    return data;
}

std::vector<double> generate_double_vector(size_t size)
{
    std::vector<double> data(size);
    std::generate(data.begin(), data.end(), []() { return generate_double(); });
    return data;
}

template<typename T>
void print_vector(std::vector<T> data)
{
    for (auto i: data)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void create_chunk()
{
    unsigned long int id = 0;
    int timestep = 1;
    std::vector<int> types = generate_int_vector(natoms);
    std::vector<double> x_cords = generate_double_vector(natoms);
    std::vector<double> y_cords = generate_double_vector(natoms);
    std::vector<double>  z_cords = generate_double_vector(natoms);
    double box_lx = generate_double();
    double box_ly = generate_double();
    double box_lz = generate_double();
    double box_xy = generate_double();
    double box_xz = generate_double();
    double box_yz = generate_double();
    // print_vector<double>(x_cords);

    boost_chunk = new BoostChunk(id,
                        timestep,
                        types,
                        x_cords,
                        y_cords,
                        z_cords,
                        box_lx,
                        box_ly,
                        box_lz,
                        box_xy,
                        box_xz,
                        box_yz);

    cereal_chunk = new CerealChunk(id,
                        timestep,
                        types,
                        x_cords,
                        y_cords,
                        z_cords,
                        box_lx,
                        box_ly,
                        box_lz,
                        box_xy,
                        box_xz,
                        box_yz);

    /*yas_chunk = new YasChunk(id,
                        timestep,
                        types,
                        x_cords,
                        y_cords,
                        z_cords,
                        box_lx,
                        box_ly,
                        box_lz,
                        box_xy,
                        box_xz,
                        box_yz);*/

}

void boost_binary_serialization()
{
    std::cout << "---===== Boost Binary Serialization" << std::endl;
    TimeVar s_start = timeNow();
    std::string serial_str;
    boost::iostreams::back_insert_device<std::string> inserter(serial_str);
    boost::iostreams::stream<boost::iostreams::back_insert_device<std::string> > s(inserter);
    boost::archive::binary_oarchive oa(s);
    oa << boost_chunk;
    s.flush();
    std::cout<<"Binary Serialization successful (size : "<<serial_str.size()<<")"<<std::endl;
    DurationMilli serialize_time_ms = timeNow() - s_start;
    std::cout << "binary_serialize_time_ms : " << serialize_time_ms.count() << std::endl;

    TimeVar d_start = timeNow();
    boost::iostreams::basic_array_source<char> device(serial_str.data(), serial_str.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > sout(device);
    boost::archive::binary_iarchive ia(sout);
    BoostChunk *new_chunk;
    ia >> new_chunk;
    DurationMilli deserialize_time_ms = timeNow() - d_start;
    std::cout << "binary_deserialize_time_ms : " << deserialize_time_ms.count() << std::endl;
    //delete new_chunk;
}

void boost_text_serialization()
{
    std::cout << "---===== Boost Binary Serialization" << std::endl;
    TimeVar s_start = timeNow();
    std::stringstream ss;
    boost::archive::text_oarchive oa(ss);
    oa << boost_chunk;
    std::string serial_str = ss.str();
    std::cout<<"Text Serialization successful (size : "<<serial_str.size()<<")"<<std::endl;
    DurationMilli serialize_time_ms = timeNow() - s_start;
    std::cout << "text_serialize_time_ms : " << serialize_time_ms.count() << std::endl;

    TimeVar d_start = timeNow();
    boost::archive::text_iarchive ia(ss);
    BoostChunk* new_chunk;
    ia >> new_chunk;
    DurationMilli deserialize_time_ms = timeNow() - d_start;
    std::cout << "text_deserialize_time_ms : " << deserialize_time_ms.count() << std::endl;

    // MDChunk *plmdchunk = dynamic_cast<MDChunk *>(new_chunk);
    // plmdchunk->print();
    delete new_chunk;
}

void cereal_binary_serialization()
{
    std::cout << "---===== Cereal Binary Serialization" << std::endl;
    TimeVar s_start = timeNow();
    std::ostringstream oss;
    cereal::BinaryOutputArchive oa(oss);
    oa(*cereal_chunk);
    std::string serial_str = oss.str();
    std::cout<<"Binary Serialization successful (size : "<<serial_str.size()<<")"<<std::endl;
    DurationMilli serialize_time_ms = timeNow() - s_start;
    std::cout << "binary_serialize_time_ms : " << serialize_time_ms.count() << std::endl;

    TimeVar d_start = timeNow();
    std::stringstream ss(serial_str);
    cereal::BinaryInputArchive ia(ss);
    CerealChunk new_chunk;
    ia(new_chunk);
    DurationMilli deserialize_time_ms = timeNow() - d_start;
    std::cout << "binary_deserialize_time_ms : " << deserialize_time_ms.count() << std::endl;

    //delete new_chunk;
}
/*
void yas_binary_serialization()
{
    std::cout << "---===== Yas Binary Serialization" << std::endl;
    TimeVar s_start = timeNow();
    constexpr std::size_t binary_opts = yas::binary|yas::elittle|yas::compacted;
    yas::mem_ostream os;
    yas::binary_oarchive<yas::mem_ostream, binary_opts> oa(os);
    oa & *yas_chunk;
    auto buf = os.get_intrusive_buffer();
    std::string serial_str;
    serial_str.assign(buf.data, buf.size);
    std::cout<<"Binary Serialization successful (size : "<<serial_str.size()<<")"<<std::endl;
    DurationMilli serialize_time_ms = timeNow() - s_start;
    std::cout << "binary_serialize_time_ms : " << serialize_time_ms.count() << std::endl;

    TimeVar d_start = timeNow();
    yas::mem_istream is(serial_str.c_str(), serial_str.size());
    yas::binary_iarchive<yas::mem_istream, binary_opts> ia(is);
    YasChunk new_chunk;
    ia & new_chunk;
    DurationMilli deserialize_time_ms = timeNow() - d_start;
    std::cout << "binary_deserialize_time_ms : " << deserialize_time_ms.count() << std::endl;
    //delete new_chunk;
}
*/
int main(int argc, char **argv)
{
    MPI_Init(NULL,NULL);
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: ./test_serialization natoms\n"); 
        return -1;
    }
    natoms = std::stoi(argv[1]);
    std::cout << "natoms = " << natoms << std::endl;
    create_chunk();
    boost_binary_serialization();
    boost_text_serialization();
    cereal_binary_serialization();
    // yas_binary_serialization();
    delete boost_chunk;
    delete cereal_chunk;
    // delete yas_chunk;
    MPI_Finalize();
    return 0;
}
