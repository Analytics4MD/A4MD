#include "dataspaces_writer.h"
#include "dataspaces.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp> 
#include <boost/serialization/vector.hpp>
#include <boost/align/align.hpp>
#include <boost/align/is_aligned.hpp>
#include <sstream>
//#include <TAU.h>

DataSpacesWriter::DataSpacesWriter(char* var_name, unsigned long int total_chunks, MPI_Comm comm)
: m_size_var_name("chunk_size"),
  m_var_name(var_name),
  m_total_chunks(total_chunks)
//  m_total_data_write_time_ms(0.0),
//  m_total_chunk_write_time_ms(0.0),
//  m_total_writer_idle_time_ms(0.0)
{
    //m_step_chunk_write_time_ms = new double[m_total_chunks];
    //m_step_writer_idle_time_ms = new double[m_total_chunks];
    m_gcomm = comm;
    MPI_Barrier(m_gcomm);
    int nprocs;
    MPI_Comm_size(m_gcomm, &nprocs);
    // Initalize DataSpaces
    // # of Peers, Application ID, ptr MPI comm, additional parameters
    // # Peers: Number of connecting clienchunk_id to the DS server
    // Application ID: Unique idenitifier (integer) for application
    // Pointer to the MPI Communicator, allows DS Layer to use MPI barrier func
    // Addt'l parameters: Placeholder for future argumenchunk_id, currently NULL.
    dspaces_init(nprocs, 1, &m_gcomm, NULL);
    printf("---===== Initialized dspaces client in DataSpacesWriter, var_name: %s, total_chunks: %u\n",m_var_name.c_str(), m_total_chunks);
}

static inline std::size_t round_up_8(std::size_t n)
{
    return (n%8 == 0) ? n : (n/8 + 1)*8;
}

void DataSpacesWriter::write_chunks(std::vector<Chunk*> chunks)
{
    //TimeVar t_start = timeNow();
    unsigned long int chunk_id; 
    MPI_Barrier(m_gcomm);
    //printf("Printing chunk before serializing\n");
    for(Chunk* chunk:chunks)
    {
        //chk_ary.print();
        //chunk->print();
        
        SerializableChunk serializable_chunk = SerializableChunk(chunk);
        // ToDo: May don't need alignment, only rounding up via padding
        //std::size_t align_size = 64;    
        //std::size_t request_size = sizeof(SerializableChunk) + align_size;
        //void *alloc = ::operator new(request_size); 
        //printf("Old allocated address: %p\n", (void*)alloc);
        //boost::alignment::align(align_size, sizeof(SerializableChunk), alloc, request_size);
        //if (boost::alignment::is_aligned(alloc, align_size))
        //{
        //    printf("New aligned address: %p\n", (void*)alloc);
        //}
        //SerializableChunk* serializable_chunk = reinterpret_cast<SerializableChunk*>(alloc);
        //*serializable_chunk = SerializableChunk(chunk);

        std::ostringstream oss;
        {
            boost::archive::text_oarchive oa(oss);
            // write class instance to archive
            oa << serializable_chunk;
            //oa << *serializable_chunk;
        }
        int ndim = 1;
        uint64_t lb[1] = {0}, ub[1] = {0};
        chunk_id = chunk->get_chunk_id();
        std::string data = oss.str();
        std::size_t size = data.length();
        //printf("MAX SIZE of string is %zu \n", data.max_size());
        //printf("chunk size for chunk_id %i is %zu\n",chunk_id,size);
       
        // Padding to multiple of 8 byte
        std::size_t c_size = round_up_8(size);
        char *c_data = new char [c_size];
        strncpy(c_data, data.c_str(), size);
        //std::size_t r_size = data.copy(c_data, size, 0);
        printf("Padded chunk size %zu\n", c_size);
        //printf("Copied chunk size %zu\n", r_size);

        m_total_size += c_size;
        //TimeVar t_istart = timeNow();
        //TAU_STATIC_TIMER_START("total_write_idle_time");
        //TAU_DYNAMIC_TIMER_START("write_idle_time");
        dspaces_lock_on_write("size_lock", &m_gcomm);
        //TAU_DYNAMIC_TIMER_STOP("write_idle_time");
        //TAU_STATIC_TIMER_STOP("total_write_idle_time");
        //DurationMilli writer_idle_time_ms = timeNow()-t_istart;
        //m_step_writer_idle_time_ms[chunk_id] = writer_idle_time_ms.count();
        //m_total_writer_idle_time_ms += m_step_writer_idle_time_ms[chunk_id];
        //TimeVar t_wstart = timeNow();
        //TAU_STATIC_TIMER_START("total_write_size_time");
        //TAU_DYNAMIC_TIMER_START("write_size_time");
        int error = dspaces_put(m_size_var_name.c_str(),
                                chunk_id,
                                sizeof(std::size_t),
                                ndim,
                                lb,
                                ub,
                                &c_size);

       
        if (error != 0)
            printf("----====== ERROR: Did not write size of chunk id: %i to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        error = dspaces_put_sync();
        if (error != 0) 
            printf("----====== ERROR: dspaces_put_sync(%s) failed\n", m_size_var_name.c_str());
        //TAU_DYNAMIC_TIMER_STOP("write_size_time");
        //TAU_STATIC_TIMER_STOP("total_write_size_time");
        
        dspaces_unlock_on_write("size_lock", &m_gcomm);
        //printf("writing char array to dataspace:\n %s\n",data.c_str());
        //TAU_STATIC_TIMER_START("total_between_write_time");
        //TAU_DYNAMIC_TIMER_START("between_write_time");
        dspaces_lock_on_write("my_test_lock", &m_gcomm);
        //TAU_DYNAMIC_TIMER_STOP("between_write_time");
        //TAU_STATIC_TIMER_STOP("total_between_write_time");
        
        //TAU_STATIC_TIMER_START("total_write_chunk_time");
        //TAU_DYNAMIC_TIMER_START("write_chunk_time");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
        error = dspaces_put(m_var_name.c_str(),
                            chunk_id,
                            c_size,
                            ndim,
                            lb,
                            ub,
                            c_data);
        if (error != 0)
            printf("----====== ERROR: Did not write chunk id: %i to dataspaces successfully\n",chunk_id);
        //else
        //   printf("Wrote char array of length %i for chunk id %i to dataspaces successfull\n",data.length(), chunk_id);
        error = dspaces_put_sync();
        if (error != 0)
            printf("----====== ERROR: dspaces_put_sync(%s) failed\n", m_var_name.c_str());
        //TAU_DYNAMIC_TIMER_STOP("write_chunk_time");
        //TAU_STATIC_TIMER_STOP("total_write_chunk_time");
        //DurationMilli write_chunk_time_ms = timeNow()-t_wstart;
        //m_step_chunk_write_time_ms[chunk_id] = write_chunk_time_ms.count();
        //printf("Chunk %lu : step_write_chunk_time_ms : %f\n", m_step_chunk_write_time_ms[chunk_id]);
        //m_total_chunk_write_time_ms += m_step_chunk_write_time_ms[chunk_id];
        dspaces_unlock_on_write("my_test_lock", &m_gcomm);
        delete[] c_data;
        //ToDo: better way to free memory of chunk
        //delete chunk;
    }
    //MPI_Barrier(m_gcomm);
    //DurationMilli write_time_ms = timeNow()-t_start;
    //m_total_data_write_time_ms += write_time_ms.count();
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_chunk_data_written : %u\n",m_total_size);
    //    printf("total_data_write_time_ms : %f\n",m_total_data_write_time_ms);
    //    printf("total_chunk_write_time_ms : %f\n",m_total_chunk_write_time_ms);
    //    printf("total_writer_idle_time_ms : %f\n",m_total_writer_idle_time_ms);
    //    printf("total_chunks written : %u\n",m_total_chunks);
    //    printf("step_chunk_write_time_ms : ");
    //    for (auto step = 0; step < m_total_chunks; step++)
    //    {
    //        printf(" %f ", m_step_chunk_write_time_ms[step]);
    //    }
    //    printf("\n");
    //    printf("step_writer_idle_time_ms : ");
    //    for (auto step = 0; step < m_total_chunks; step++)
    //    {
    //        printf(" %f ", m_step_writer_idle_time_ms[step]);
    //    }
    //    printf("\n");
    //    //ToDo: delete in destructor
    //    delete[] m_step_chunk_write_time_ms;
    //    delete[] m_step_writer_idle_time_ms;
    }
}

DataSpacesWriter::~DataSpacesWriter() 
{
    MPI_Barrier(m_gcomm);
    dspaces_finalize();
    printf("---===== Finalized dspaces client in DataSpacesWriter\n");
}

