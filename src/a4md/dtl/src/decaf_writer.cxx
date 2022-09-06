#include "../include/decaf_writer.h"

#include "../include/chunk_serializer.h"
#ifdef BUILT_IN_PERF
#include "../../common/include/timer.h"
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
    m_step_data_write_time_ms = new double[m_total_chunks];
    m_step_chunk_write_time_ms = new double[m_total_chunks];
    m_step_writer_idle_time_ms = new double[m_total_chunks];
    m_step_ser_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    Workflow workflow;
    Workflow::make_wflow_from_json(workflow, m_json_file.c_str());
    m_decaf = new decaf::Decaf(m_gcomm, workflow);
    printf("---===== Initialized DecafWriter with json_file: %s, total_chunks: %u\n", m_json_file.c_str(), m_total_chunks);
}

DecafWriter::DecafWriter(decaf::Decaf* decaf, unsigned long int total_chunks, MPI_Comm comm)
: m_decaf(decaf),
#ifdef BUILT_IN_PERF
  m_total_data_write_time_ms(0.0),
  m_total_chunk_write_time_ms(0.0),
  m_total_writer_idle_time_ms(0.0),
  m_total_ser_time_ms(0.0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_data_write_time_ms = new double[m_total_chunks];
    m_step_chunk_write_time_ms = new double[m_total_chunks];
    m_step_writer_idle_time_ms = new double[m_total_chunks];
    m_step_ser_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    initialized = true;
    printf("---===== Initialized DecafWriter on initialized Decaf node, total_chunks: %u\n", m_total_chunks);
}

static inline std::size_t round_up_8(std::size_t n)
{
    return (n%8 == 0) ? n : (n/8 + 1)*8;
}

void DecafWriter::write_chunks(std::vector<Chunk*> chunks)
{
#ifdef TAU_PERF
    TAU_STATIC_TIMER_START("total_write_time");
    TAU_DYNAMIC_TIMER_START("step_write_time");
#endif  
#ifdef BUILT_IN_PERF
    TimeVar t_start = timeNow();
#endif
    unsigned long int chunk_id; 
    printf("---===== DecafWriter::write_chunks\n");
    // MPI_Barrier(m_gcomm);
    for(Chunk* chunk:chunks)
    {
        chunk_id = chunk->get_chunk_id();

        /* ----- Boost Binary Serialization ----- */
#ifdef BUILT_IN_PERF
        TimeVar t_serstart = timeNow();
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
            printf("----====== ERROR: DecafWriter::write_chunks Failed to serialize chunk\n");
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
#ifdef BUILT_IN_PERF
        DurationMilli ser_time_ms = timeNow() - t_serstart;
        m_step_ser_time_ms[chunk_id] = ser_time_ms.count();
        m_total_ser_time_ms += m_step_ser_time_ms[chunk_id];
#endif

        printf("---===== DecafWriter::write_chunks Chunk size %zu\n", c_size);
        m_total_size += c_size;

#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_idle_time");
        TAU_DYNAMIC_TIMER_START("step_write_idle_time");
#endif

#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_idle_time");
        TAU_STATIC_TIMER_STOP("total_write_idle_time");
#endif

#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_write_chunk_time");
        TAU_DYNAMIC_TIMER_START("step_write_chunk_time");
#endif
        decaf::ArrayFieldc field_data(c_data, c_size, c_size);
        decaf::pConstructData container;
        container->appendData("chunk", field_data, decaf::DECAF_NOFLAG, decaf::DECAF_PRIVATE, decaf::DECAF_SPLIT_DEFAULT, decaf::DECAF_MERGE_DEFAULT);
        while (!m_decaf->put(container)){
        } 
        printf("---===== DecafWriter::write_chunks Successfully put chunk: %lu\n", chunk_id);

#ifdef BUILT_IN_PERF
        TimeVar t_wcend = timeNow();
        TimeVar t_iend = (m_decaf->get_out_mids()).front();
        DurationMilli writer_idle_time_ms = t_iend-t_istart;
        m_step_writer_idle_time_ms[chunk_id] = writer_idle_time_ms.count();
        m_total_writer_idle_time_ms += m_step_writer_idle_time_ms[chunk_id];

        DurationMilli write_chunk_time_ms = t_wcend-t_iend;
        m_step_chunk_write_time_ms[chunk_id] = write_chunk_time_ms.count();
        m_total_chunk_write_time_ms += m_step_chunk_write_time_ms[chunk_id];
        
#endif
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_write_chunk_time");
        TAU_STATIC_TIMER_STOP("total_write_chunk_time");  
#endif

// #ifdef NERSC
//         delete[] c_data;
// #endif

#ifdef COUNT_LOST_FRAMES   

#endif    
    }
    //MPI_Barrier(m_gcomm);

#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_write_time");
    TAU_STATIC_TIMER_STOP("total_write_time");
#endif
#ifdef BUILT_IN_PERF
    DurationMilli write_time_ms = timeNow()-t_start;
    m_step_data_write_time_ms[chunk_id] = write_time_ms.count();
    m_total_data_write_time_ms += m_step_data_write_time_ms[chunk_id];
#endif

#ifdef BUILT_IN_PERF
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_chunks written : %u\n",m_total_chunks);
        printf("total_chunk_data_written : %u\n",m_total_size);
        printf("total_data_write_time_ms : %f\n",m_total_data_write_time_ms);
        printf("total_chunk_write_time_ms : %f\n",m_total_chunk_write_time_ms);
        printf("total_writer_idle_time_ms : %f\n",m_total_writer_idle_time_ms);
        printf("total_ser_time_ms : %f\n",m_total_ser_time_ms);
        printf("step_data_write_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_data_write_time_ms[step]);
        }
        printf("\n");
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
        printf("step_ser_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_ser_time_ms[step]);
        }
        printf("\n");

        //Free Built-in Performance Variables
        delete[] m_step_data_write_time_ms;
        delete[] m_step_chunk_write_time_ms;
        delete[] m_step_writer_idle_time_ms;
        delete[] m_step_ser_time_ms;
    }
#endif
}

DecafWriter::~DecafWriter() 
{
    // MPI_Barrier(m_gcomm);
    if (!initialized) {
        printf("Terminating decaf\n");
        m_decaf->terminate();
        delete m_decaf;
    }
    printf("---===== Finalized DecafWriter\n");
}
