#include "../include/chunker.h"
#include "../include/pycall.h"

Chunker::~Chunker()
{
    printf("---===== Finalized Chunker\n");
}

std::vector<Chunk*> Chunker::get_chunks(int num_chunks)
{
    if (m_chunkq.size() < num_chunks)
        throw "get_chunks asking for more chunks than what is available";
    std::vector<Chunk*> chunks;
    for (int i=0; i<num_chunks; i++)
    {
        // Don't we need latest chunks ???
        chunks.push_back(m_chunkq.front());
        // Should delete element after accessing ???
        m_chunkq.pop();
    }
    return chunks;
}

void Chunker::append_chunk(Chunk* chunk)
{
    // ToDo: buffer size, check if buffer is full
    m_chunkq.push(chunk);
    m_next_id += 1;
}
