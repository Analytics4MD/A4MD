#include "../include/chunk_stager.h"
#include <vector>

ChunkStager::ChunkStager(ChunkReader* chunk_reader, std::vector<ChunkOperator*> chunk_operators, ChunkWriter* chunk_writer)
: m_chunk_reader(chunk_reader),
  m_chunk_writer(chunk_writer)
{
    m_chunk_operators = chunk_operators;
    printf("---===== Initalized ChunkStager\n");
}

ChunkStager::~ChunkStager()
{
    printf("---===== Finalized ChunkStager\n");
}

bool ChunkStager::stage_chunks(int num_chunks)
{
    bool success = false;
    throw NotImplementedException("Need to refactor chunkstager to use from and to chunk if instead of num chunks.");
    //auto chunks = m_chunk_reader.read_chunks(num_chunks);
    //success = m_chunk_writer.write_chunks(chunks);
    //throw NotImplementedException();
    return success;
}

void ChunkStager::stage_chunks(unsigned long int chunk_id_from, unsigned long int chunk_id_to)
{
    printf("---===== ChunkStager::stage_chunks() --> Stage chunks from %lu to %lu\n", chunk_id_from, chunk_id_to);
    for (unsigned long int chunk_id = chunk_id_from; chunk_id <= chunk_id_from; chunk_id++)
    {
        // Read chunks by ChunkReader
        std::vector<Chunk*> input_chunks = m_chunk_reader->read_chunks(chunk_id, chunk_id);

        //Operate chunks by single/multiple ChunkOperators
        std::vector<Chunk*> output_chunks;
        for (ChunkOperator* chunk_operator : m_chunk_operators) {
            output_chunks = chunk_operator->operate_chunks(input_chunks);
            free_chunks(input_chunks);
            input_chunks = output_chunks;
        }

        // Write chunks by ChunkWriter
        if (m_chunk_writer != nullptr)
        {
            m_chunk_writer->write_chunks(input_chunks);
            free_chunks(input_chunks);
        }
    }
}
