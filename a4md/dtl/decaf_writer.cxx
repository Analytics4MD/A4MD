#include "decaf_writer.h"
#include <bredala/data_model/pconstructtype.h>
#include <bredala/data_model/arrayfield.hpp>
#include <bredala/data_model/boost_macros.h>
#ifdef TRANSPORT_CCI
#include <cci.h>
#endif
#include "chunk_serializer.h"
#ifdef BUILT_IN_PERF
#include "timer.h"
#endif
#ifdef TAU_PERF
#include <TAU.h>
#endif

DecafWriter::DecafWriter(std::string json_file, unsigned long int total_chunks, MPI_Comm comm)
: m_json_file(json_file),
#ifdef BUILT_IN_PERF
  m_total_data_write_time_ms(0.0),
  m_total_chunk_write_time_ms(0.0),
  m_total_writer_idle_time_ms(0.0),
  m_total_ser_time_ms(0.0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_chunk_write_time_ms = new double[m_total_chunks];
    m_step_writer_idle_time_ms = new double[m_total_chunks];
    m_step_size_write_time_ms = new double[m_total_chunks];
    m_step_between_write_time_ms = new double[m_total_chunks];
    m_step_ser_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    Workflow workflow;
    Workflow::make_wflow_from_json(workflow, m_json_file.c_str());
    printf("Initializing decaf\n");
    m_decaf = new decaf::Decaf(m_gcomm, workflow);
    printf("---===== Initialized DecafWriter with json_file: %s, total_chunks: %u\n", m_json_file.c_str(), m_total_chunks);
}

static inline std::size_t round_up_8(std::size_t n)
{
    return (n%8 == 0) ? n : (n/8 + 1)*8;
}

void DecafWriter::write_chunks(std::vector<Chunk*> chunks)
{
#ifdef BUILT_IN_PERF
    TimeVar t_start = timeNow();
#endif
    unsigned long int chunk_id; 
    printf("---===== DecafReader::write_chunks\n");
    // MPI_Barrier(m_gcomm);
    for(Chunk* chunk:chunks)
    {
        chunk_id = chunk->get_chunk_id();

        /* ----- Boost Binary Serialization ----- */
#ifdef BUILT_IN_PERF
        TimeVar t_serstart = timeNow();
#endif       
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_time");
        TAU_DYNAMIC_TIMER_START("step_write_time");
#endif   
#ifdef TAU_PERF     
        TAU_STATIC_TIMER_START("total_write_ser_time");
        TAU_DYNAMIC_TIMER_START("step_write_ser_time");
#endif 
        SerializableChunk serializable_chunk = SerializableChunk(chunk); 
        std::string data;
        ChunkSerializer<SerializableChunk> chunk_serializer;
        bool ret = chunk_serializer.serialize(serializable_chunk, data);
        if (!ret)
        {
            printf("----====== ERROR: Failed to serialize chunk\n");
        }
        std::size_t size = data.size();

#ifdef NERSC
        // Padding to multiple of 8 byte
        std::size_t c_size = round_up_8(size);
        char *c_data = new char [c_size];
        //strncpy(c_data, data.c_str(), size);
        std::memcpy(c_data, data.c_str(), size);
        // printf("Padded chunk size %zu\n", c_size);
#else
        std::size_t c_size = size;
        char *c_data = (char*)data.data();
#endif /* NERSC */

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_ser_time");
        TAU_STATIC_TIMER_STOP("total_write_ser_time");
#endif
#ifdef TAU_PERF     
        TAU_DYNAMIC_TIMER_STOP("step_write_time");
        TAU_STATIC_TIMER_STOP("total_write_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli ser_time_ms = timeNow() - t_serstart;
        m_step_ser_time_ms[chunk_id] = ser_time_ms.count();
        m_total_ser_time_ms += m_step_ser_time_ms[chunk_id];
#endif

        // vector<int> types = chunk->get_types();
        // vector<double> x_positions = chunk->get_x_positions();
        // vector<double> y_positions = chunk->get_y_positions();
        // vector<double> z_positions = chunk->get_z_positions();
        // double box_lx = chunk->get_box_lx();
        // double box_ly = chunk->get_box_ly();
        // double box_lz = chunk->get_box_lz();
        // double box_xy = chunk->get_box_xy();
        // double box_xz = chunk->get_box_xz();
        // double box_yz = chunk->get_box_yz();
        // int timestep = chunk->get_timestep();
        
        // VectorFieldi field_types(types, 1);
        // VectorFliedd field_x_positions(x_positions,1);
        // VectorFliedd field_y_positions(y_positions,1);
        // VectorFliedd field_z_positions(z_positions,1);
        // SimpleFieldd field_box_lx(box_lx);
        // SimpleFieldd field_box_ly(box_ly);
        // SimpleFieldd field_box_lz(box_lz);
        // SimpleFieldd field_box_xy(box_xy);
        // SimpleFieldd field_box_xz(box_xz);
        // SimpleFieldd field_box_yz(box_yz);
        // SimpleFieldi field_timestep(timestep);

        // pConstructData container;
        // container->appendData("types", field_types, DECAF_NOFLAG, DECAF_PRIVATE, DECAF_SPLIT_DEFAULT, DECAF_MERGE_DEFAULT);
        // container->appendData("x_position", field_x_positions, DECAF_NOFLAG, DECAF_PRIVATE, DECAF_SPLIT_DEFAULT, DECAF_MERGE_DEFAULT);
        // container->appendData("y_position", field_x_positions, DECAF_NOFLAG, DECAF_PRIVATE, DECAF_SPLIT_DEFAULT, DECAF_MERGE_DEFAULT);
        // container->appendData("z_position", field_y_positions, DECAF_NOFLAG, DECAF_PRIVATE, DECAF_SPLIT_DEFAULT, DECAF_MERGE_DEFAULT);
        // container->appendData("timestep", field_timestep, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);
        // container->appendData("box_lx", field_box_lx, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);
        // container->appendData("box_ly", field_box_ly, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);
        // container->appendData("box_lz", field_box_lz, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);
        // container->appendData("box_xy", field_box_xy, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);
        // container->appendData("box_xz", field_box_xz, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);
        // container->appendData("box_yz", field_box_yz, DECAF_NOFLAG, DECAF_SHARED, DECAF_SPLIT_KEEP_VALUE, DECAF_MERGE_DEFAULT);

        printf("Chunk size %zu\n", c_size);
        m_total_size += c_size;

        decaf::ArrayFieldc field_data(c_data, c_size, c_size);
        decaf::pConstructData container;
        container->appendData("chunk", field_data, decaf::DECAF_NOFLAG, decaf::DECAF_PRIVATE, decaf::DECAF_SPLIT_DEFAULT, decaf::DECAF_MERGE_DEFAULT);
        m_decaf->put(container);
        

#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_stall_time");
        TAU_DYNAMIC_TIMER_START("step_write_stall_time");
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_idle_time");
        TAU_DYNAMIC_TIMER_START("step_write_idle_time");
#endif
        
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_idle_time");
        TAU_STATIC_TIMER_STOP("total_write_idle_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli writer_idle_time_ms = timeNow()-t_istart;
        m_step_writer_idle_time_ms[chunk_id] = writer_idle_time_ms.count();
        m_total_writer_idle_time_ms += m_step_writer_idle_time_ms[chunk_id];
        TimeVar t_wsstart = timeNow();
#endif

#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_size_time");
        TAU_DYNAMIC_TIMER_START("step_write_size_time");
#endif


#ifdef BUILT_IN_PERF
        DurationMilli size_write_time_ms = timeNow() - t_wsstart;
        m_step_size_write_time_ms[chunk_id] = size_write_time_ms.count();
#endif

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_size_time");
        TAU_STATIC_TIMER_STOP("total_write_size_time");
#endif
        
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_between_time");
        TAU_DYNAMIC_TIMER_START("step_write_between_time");
#endif

#ifdef BUILT_IN_PERF
        TimeVar t_wbstart = timeNow();
#endif

#ifdef BUILT_IN_PERF
        DurationMilli between_write_time_ms = timeNow() - t_wbstart; 
        m_step_between_write_time_ms[chunk_id] = between_write_time_ms.count();
        TimeVar t_wcstart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_between_time");
        TAU_STATIC_TIMER_STOP("total_write_between_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_write_stall_time");
        TAU_STATIC_TIMER_STOP("total_write_stall_time");
        
        TAU_STATIC_TIMER_START("total_write_time");
        TAU_DYNAMIC_TIMER_START("step_write_time");
        
        TAU_STATIC_TIMER_START("total_write_chunk_time");
        TAU_DYNAMIC_TIMER_START("step_write_chunk_time");
        //TAU_TRACK_MEMORY_FOOTPRINT();
        //TAU_TRACK_MEMORY_FOOTPRINT_HERE();
#endif
        
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_chunk_time");
        TAU_STATIC_TIMER_STOP("total_write_chunk_time");
        
        TAU_DYNAMIC_TIMER_STOP("step_write_time");
        TAU_STATIC_TIMER_STOP("total_write_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli write_chunk_time_ms = timeNow()-t_wcstart;
        m_step_chunk_write_time_ms[chunk_id] = write_chunk_time_ms.count();
        m_total_chunk_write_time_ms += m_step_chunk_write_time_ms[chunk_id];
#endif

// #ifdef NERSC
//         delete[] c_data;
// #endif

#ifdef COUNT_LOST_FRAMES   

#endif    
    }
    //MPI_Barrier(m_gcomm);
#ifdef BUILT_IN_PERF
    DurationMilli write_time_ms = timeNow()-t_start;
    m_total_data_write_time_ms += write_time_ms.count();
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_chunks written : %u\n",m_total_chunks);
        printf("total_chunk_data_written : %u\n",m_total_size);
        printf("total_data_write_time_ms : %f\n",m_total_data_write_time_ms);
        printf("total_chunk_write_time_ms : %f\n",m_total_chunk_write_time_ms);
        printf("total_writer_idle_time_ms : %f\n",m_total_writer_idle_time_ms);
        printf("total_ser_time_ms : %f\n",m_total_ser_time_ms);
        printf("step_chunk_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_chunk_write_time_ms[step]);
        }
        printf("\n");
        printf("step_writer_idle_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_writer_idle_time_ms[step]);
        }
        printf("\n");
        printf("step_size_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_size_write_time_ms[step]);
        }
        printf("\n");
        printf("step_between_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_between_write_time_ms[step]);
        }
        printf("\n");
        printf("step_ser_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_ser_time_ms[step]);
        }
        printf("\n");

        //Free Built-in Performance Variables
        delete[] m_step_chunk_write_time_ms;
        delete[] m_step_writer_idle_time_ms;
        delete[] m_step_size_write_time_ms;
        delete[] m_step_between_write_time_ms;
        delete[] m_step_ser_time_ms;
    }
#endif
}

DecafWriter::~DecafWriter() 
{
    // MPI_Barrier(m_gcomm);
    printf("Terminating decaf\n");
    m_decaf->terminate();
    delete m_decaf;
    printf("---===== Finalized DecafWriter\n");
}
