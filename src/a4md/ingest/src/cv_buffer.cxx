#include "../include/cv_buffer.h"

CVBuffer::CVBuffer(int interval)
{
	m_interval = interval;
	m_buffer_id = 0;
	printf("---===== Initialized CVBuffer with interval = %d\n", m_interval);
}

CVBuffer::~CVBuffer()
{
	printf("---===== Finalized CVBuffer\n");
}

std::vector<Chunk*> CVBuffer::operate_chunks(std::vector<Chunk*> chunks)
{
	printf("---===== CVBuffer::operate_chunks at buffer_id = %d\n", m_buffer_id);
	std::vector<Chunk*> result;
	if (m_buffer_id % m_interval == 0) 
	{
		m_chunk_buffer = new CVChunk(m_buffer_id / m_interval);
	}

	for(auto chunk : chunks)
	{
		m_chunk_buffer->append(chunk);
	}

	if ((m_buffer_id + 1) % m_interval == 0)
	{
		result.push_back(m_chunk_buffer);
	}
	m_buffer_id++;

	return result;
}
