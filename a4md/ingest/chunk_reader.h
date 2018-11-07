#ifndef __CHUNK_READER_H__
#define __CHUNK_READER_H__

class ChunkReader {
private:
	Chunk chunk;
public:
	virtual void get_chunk();
}
#endif