import faiss
# Memory System
class MemorySystem:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.memory_data = []

    def add_memory(self, embedding, data):
        embedding_np = embedding.detach().cpu().numpy()
        self.index.add(embedding_np)
        self.memory_data.append(data)

    def retrieve_memory(self, query_embedding, k=5):
        if self.index.ntotal == 0:
            return []
        query_np = query_embedding.detach().cpu().numpy()
        distances, indices = self.index.search(query_np, k)
        retrieved_data = [self.memory_data[i] for i in indices[0]]
        return retrieved_data
