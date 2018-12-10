#ifndef __IMS_WRITER_H__
#define __IMS_WRITER_H__
#include "chunk.h"
#include <chrono>
#define timeNow() std::chrono::high_resolution_clock::now()
typedef std::chrono::high_resolution_clock::time_point TimeVar;
typedef std::chrono::duration<double, std::milli> DurationMilli;


// Writes chunks into an IMS. No application logic here.
class IMSWriter 
{
    protected:
        double m_total_data_write_time_ms = 0.0;
    public:
        virtual void write_chunks(std::vector<Chunk*> chunks);
};
#endif
