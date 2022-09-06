#ifndef __MD_INGEST_H__
#define __MD_INGEST_H__
#include "ingest.h"
#include "chunk_stager.h"

class MDIngest : public Ingest
{
    private:
        std::string m_file_path;
        std::string m_log_path;
        std::string m_py_path;
        std::string m_py_script;
        std::string m_py_def;

    protected:
        std::vector<ChunkStager> get_stagers();
    public:
        MDIngest(int max_chunk_id,
                 std::string file_path = "mstring.pdb",
                 std::string log_path = "temp_log",
                 std::string py_path = "./",
                 std::string py_script = "analysis_script.py", 
                 std::string py_def = "analyze");
    //    ~MDIngest();
};
#endif
