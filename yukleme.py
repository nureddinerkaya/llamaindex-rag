from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_connection import vector_store

# 0. Embed Model
embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B")


# The logic that used to run at import time (load, split, embed, add) is now in a function

def build_nodes_from_file(file_path: str = "./data/llama2.pdf"):
    """Load a file, split into text chunks, construct TextNode objects and return the node list.

    This function does NOT add nodes to the vector store; it only returns them so the caller
    can decide when/how to add (prevents accidental duplicate inserts on import).
    """
    # 1. Load Data
    loader = PyMuPDFReader()
    documents = loader.load(file_path=file_path)

    # 2. Use a Text Splitter to Split Documents
    text_parser = SentenceSplitter(
        chunk_size=1024,
        # separator=" ",
    )

    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    # 3. Manually Construct Nodes from Text Chunks
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    # 4. Generate Embeddings for each Node
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content()
        )
        node.embedding = node_embedding

    return nodes


def upload_nodes_from_file(file_path: str = "./data/llama2.pdf"):
    """Build nodes from a file and add them to the configured vector store.

    Call this function explicitly when you want to index/upload documents. Importing
    `embed_model` from this module will no longer trigger uploads.
    """
    nodes = build_nodes_from_file(file_path=file_path)
    # 5. Load Nodes into a Vector Store
    vector_store.add(nodes)
    return len(nodes)


if __name__ == "__main__":
    # When executed as a script, perform the upload (keeps import-time safe)
    num = upload_nodes_from_file()
    print(f"Uploaded {num} nodes to vector store.")
