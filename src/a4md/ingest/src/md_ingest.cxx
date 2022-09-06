#include "../include/md_ingest.h"

MDIngest::MDIngest(int max_chunk_id,
                   std::string file_path,
                   std::string log_path,
                   std::string py_path,
                   std::string py_script, 
                   std::string py_def) 
: Ingest(max_chunk_id)
{
    //m_file_path = file_path;
    //m_log_path = log_path;
    //m_py_path = py_path;
    //m_py_script = py_script;
    //m_py_def = py_def;
}

//TODO: free stagers in the destructor
//
//

std::vector<ChunkStager> MDIngest::get_stagers()
{
    std::vector<ChunkStager> stagers;
    
    //Chunker* pdb_chunker = new PdbChunker(m_file_path, m_log_path, m_py_path, m_py_script, m_py_def); 
    //ChunkReader* md_reader = new ChunkReader(pdb_chunker);
    //ChunkWriter* md_writer = new ChunkWriter();
    //ChunkStager chStager = ChunkStager(md_reader, md_writer);
    //stagers.push_back(chStager);

    return stagers;
}
