#ifndef __PLUMED_INGEST_H__
#define __PLUMED_INGEST_H__
#include "ingest.h"

class PlumedIngest : public Ingest
{
    protected:
        std::vector<ChunkStager> get_stagers();
};
#endif
