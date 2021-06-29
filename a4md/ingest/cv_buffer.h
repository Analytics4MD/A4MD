#ifndef __CV_BUFFER_H__
#define __CV_BUFFER_H__
#include <vector>
#include "chunk_operator.h"
#include "cv_chunk.h"

class CVBuffer : public ChunkOperator
{
    private:
        std::shared_ptr<Chunk> m_chunk_buffer;
        int m_interval;
        int m_buffer_id;
    public:
        CVBuffer(int interval);
        ~CVBuffer();
        std::vector<std::shared_ptr<Chunk>> operate_chunks(std::vector<std::shared_ptr<Chunk>> chunks) override;
};
#endif /* __CV_BUFFER_H__ */