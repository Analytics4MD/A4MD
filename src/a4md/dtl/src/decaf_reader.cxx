#include "../include/decaf_reader.h"

// #include <bredala/data_model/boost_macros.h>
#include "../include/chunk_serializer.h"
#if defined(BUILT_IN_PERF) || defined(COUNT_LOST_FRAMES)
#include "../../common/include/timer.h"
#endif
#ifdef TAU_PERF
#include <TAU.h>
#endif
#ifdef COUNT_LOST_FRAMES
#include <thread>
#endif

DecafReader::DecafReader(std::string json_file, unsigned long int total_chunks, MPI_Comm comm)
: m_json_file(json_file),
#ifdef BUILT_IN_PERF
  m_total_data_read_time_ms(0.0),
  m_total_chunk_read_time_ms(0.0),
  m_total_reader_idle_time_ms(0.0),
  m_total_deser_time_ms(0.0),
#endif
#ifdef COUNT_LOST_FRAMES
  m_wait_ms(1000),// default wait time of 1 second
  m_min_wait_ms(100),// default min wait time of 100 ms
  m_max_wait_ms(30000),//defailt max wait time of 30 seconds
  m_lost_frames_count(0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_data_read_time_ms = new double [m_total_chunks];
    m_step_chunk_read_time_ms = new double [m_total_chunks];
    m_step_reader_idle_time_ms = new double [m_total_chunks];
    m_step_deser_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    Workflow workflow;
    Workflow::make_wflow_from_json(workflow, m_json_file.c_str());
    m_decaf = new decaf::Decaf(m_gcomm, workflow);
    printf("---===== Initialized DecafReader with json_file: %s, total_chunks: %u\n", m_json_file.c_str(), m_total_chunks);
}

DecafReader::DecafReader(decaf::Decaf* decaf, unsigned long int total_chunks, MPI_Comm comm)
: m_decaf(decaf),
#ifdef BUILT_IN_PERF
  m_total_data_read_time_ms(0.0),
  m_total_chunk_read_time_ms(0.0),
  m_total_reader_idle_time_ms(0.0),
  m_total_deser_time_ms(0.0),
#endif
#ifdef COUNT_LOST_FRAMES
  m_wait_ms(1000),// default wait time of 1 second
  m_min_wait_ms(100),// default min wait time of 100 ms
  m_max_wait_ms(30000),//defailt max wait time of 30 seconds
  m_lost_frames_count(0),
#endif
  m_total_chunks(total_chunks)
{
#ifdef BUILT_IN_PERF
    m_step_data_read_time_ms = new double [m_total_chunks];
    m_step_chunk_read_time_ms = new double [m_total_chunks];
    m_step_reader_idle_time_ms = new double [m_total_chunks];
    m_step_deser_time_ms = new double[m_total_chunks];
#endif
    m_gcomm = comm;
    initialized = true; 
    printf("---===== Initialized DecafReader with initialized Decaf node, total_chunks: %u\n", m_total_chunks);
}

std::vector<Chunk*> DecafReader::read_chunks(unsigned long int chunks_from, unsigned long int chunks_to)
{
#ifdef BUILT_IN_PERF
    TimeVar t_start = timeNow();
#endif
    unsigned long int chunk_id;
    printf("---===== DecafReader::get_chunks with chunk_from %lu, chunk_to %lu\n",chunks_from, chunks_to);
    
    // MPI_Barrier(m_gcomm);
    std::vector<Chunk*> chunks; 


    // if (m_decaf->get(in_data))
    // {

    for (auto i = 0; i <= chunks_to - chunks_from; i++)
    {
        map<std::string, decaf::pConstructData> in_data;

#ifdef BUILT_IN_PERF
        TimeVar t_istart = timeNow();
#endif
#ifdef TAU_PERF
        TAU_STATIC_TIMER_START("total_read_time");
        TAU_DYNAMIC_TIMER_START("step_read_time");

        TAU_STATIC_TIMER_START("total_read_idle_time");
        TAU_DYNAMIC_TIMER_START("step_read_idle_time");
#endif
        chunk_id = chunks_from;
        while (!m_decaf->get(in_data)) {
        }
#ifdef TAU_PERF
        TAU_DYNAMIC_TIMER_STOP("step_read_idle_time");
        TAU_STATIC_TIMER_STOP("total_read_idle_time");

        TAU_STATIC_TIMER_START("total_read_chunk_time");
        TAU_DYNAMIC_TIMER_START("step_read_chunk_time");
#endif
#ifdef BUILT_IN_PERF
        DurationMilli reader_idle_time_ms = timeNow()-t_istart;
        m_step_reader_idle_time_ms[chunk_id] = reader_idle_time_ms.count();
        m_total_reader_idle_time_ms += m_step_reader_idle_time_ms[chunk_id];
        TimeVar t_rcstart = timeNow();
#endif

        decaf::ArrayFieldc field_data = in_data.at("in")->getFieldData<decaf::ArrayFieldc>("chunk");
        if (field_data) {
            printf("---===== DecafReader::get_chunks Successfully get chunk: %lu \n", chunks_from + i);
            std::size_t chunk_size = field_data.getArraySize();
            char* data = field_data.getArray();
            printf("---===== DecafReader::get_chunks Chunk size %zu\n", chunk_size);

#ifdef TAU_PERF
            TAU_DYNAMIC_TIMER_STOP("step_read_chunk_time");
            TAU_STATIC_TIMER_STOP("total_read_chunk_time");

            TAU_STATIC_TIMER_START("total_read_deser_time");
            TAU_DYNAMIC_TIMER_START("step_read_deser_time");
#endif
#ifdef BUILT_IN_PERF
            DurationMilli read_chunk_time_ms = timeNow()-t_rcstart;
            m_step_chunk_read_time_ms[chunk_id] = read_chunk_time_ms.count();
            m_total_chunk_read_time_ms += m_step_chunk_read_time_ms[chunk_id];
            TimeVar t_deserstart = timeNow();
#endif

            SerializableChunk serializable_chunk;
            ChunkSerializer<SerializableChunk> chunk_serializer;
            bool ret = chunk_serializer.deserialize(serializable_chunk, data, chunk_size);
            if (!ret)
            {
                printf("----===== ERROR: DecafReader::get_chunks Failed to deserialize chunk\n");
            }
#ifdef TAU_PERF
            TAU_DYNAMIC_TIMER_STOP("step_read_deser_time");
            TAU_STATIC_TIMER_STOP("total_read_deser_time");
#endif
#ifdef BUILT_IN_PERF
            DurationMilli deser_time_ms = timeNow() - t_deserstart;
            m_step_deser_time_ms[chunk_id] = deser_time_ms.count();
            m_total_deser_time_ms += m_step_deser_time_ms[chunk_id];
#endif
            chunks.push_back(serializable_chunk.get_chunk());
            //delete[] data;
        } else {
            printf("---===== ERROR: DecafReader::get_chunks Something went wrong while trying to get chunk: %lu \n", chunks_from + i);
        }
    }
    // }

#ifdef TAU_PERF
    TAU_DYNAMIC_TIMER_STOP("step_read_time");
    TAU_STATIC_TIMER_STOP("total_read_time");
#endif 
#ifdef BUILT_IN_PERF
    DurationMilli read_time_ms = timeNow()-t_start;
    m_step_data_read_time_ms[chunk_id] = read_time_ms.count();
    m_total_data_read_time_ms += m_step_data_read_time_ms[chunk_id];
#endif
    //MPI_Barrier(m_gcomm);

#ifdef BUILT_IN_PERF
    if (chunk_id == m_total_chunks-1)
    {
        printf("total_chunks read : %u\n",m_total_chunks);
        printf("total_data_read_time_ms : %f\n",m_total_data_read_time_ms);
        printf("total_chunk_read_time_ms : %f\n",m_total_chunk_read_time_ms);
        printf("total_reader_idle_time_ms : %f\n",m_total_reader_idle_time_ms);
        printf("total_deser_time_ms : %f\n",m_total_deser_time_ms);
        printf("step_data_read_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_data_read_time_ms[step]);
        }
        printf("\n");
        printf("step_chunk_read_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_chunk_read_time_ms[step]);
        }
        printf("\n");
        printf("step_reader_idle_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_reader_idle_time_ms[step]);
        }
        printf("\n");
        printf("step_deser_time_ms : ");
        for (auto step = 0; step < m_total_chunks; step++)
        {
            printf(" %f ", m_step_deser_time_ms[step]);
        }
        printf("\n");

        //ToDo: delete in destructor
        delete[] m_step_data_read_time_ms;
        delete[] m_step_chunk_read_time_ms;
        delete[] m_step_reader_idle_time_ms;
        delete[] m_step_deser_time_ms;
    }
#endif
    return chunks;
}

DecafReader::~DecafReader()
{
    // MPI_Barrier(m_gcomm);
    if (!initialized) {
        printf("Terminating decaf\n");
        m_decaf->terminate();
        delete m_decaf;
    }
    printf("---===== Finalized DecafReader\n");
}
