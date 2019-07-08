#ifndef __CHUNK_SERIALIZER_H__
#define __CHUNK_SERIALIZER_H__
#include <string>


template<typename SerializableContainer>
class ChunkSerializer
{
	public:
		ChunkSerializer();
		~ChunkSerializer();

		bool serialize(const SerializableContainer& serializable_container, std::string& serialized_buffer);
		bool deserialize(SerializableContainer& serializable_container, const std::string& serialized_buffer);
};

#endif /* __CHUNK_SERIALIZER_H__ */