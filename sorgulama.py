from qdrant_connection import vector_store
from yukleme import set_embed_model
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Optional

# This is example code demonstrating how to query the vector store
# Initialize embedding model
embed_model = set_embed_model("BAAI/bge-m3")

# 1. Create Query Embedding (example query)
query_str = "Can you tell me about the key concepts for safety finetuning"
query_embedding = embed_model.get_query_embedding(query_str)

# 2. Query Vector Database

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode  # type: ignore[arg-type]
)

# returns a VectorStoreQueryResult
query_result = vector_store.query(vector_store_query)
print(query_result.nodes[0].get_content())

# 3. Parse Results into a Set of Nodes
nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))
