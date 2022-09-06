#ifndef __CV_BUFFER_H__
#define __CV_BUFFER_H__
#include <vector>
#include "../../common/include/chunk_operator.h"
#include "../../common/include/cv_chunk.h"

class CVBuffer : public ChunkOperator
{
    private:
        Chunk* m_chunk_buffer;
        int m_interval;
        int m_buffer_id;
    public:
        CVBuffer(int interval);
        ~CVBuffer();
        std::vector<Chunk*> operate_chunks(std::vector<Chunk*> chunks) override;
};
#endif /* __CV_BUFFER_H__ */
