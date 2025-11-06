import os
import sys
import json
from time import perf_counter
from typing import List, Optional

from llama_index.core.schema import TextNode


def build_nodes_from_json(file_path: str, embed_model) -> List[TextNode]:
    """Read a JSON file containing a list of dicts and return a list of TextNode

    Each dictionary becomes a single TextNode. The node.text contains a
    human-readable representation of the dictionary and the original dict is
    attached to node.metadata so it is preserved when stored in the vector DB.
    Embeddings are NOT generated in this function. The caller should pass a
    configured embed_model if it wants to generate embeddings in-place.
    """
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the FAQ JSON to be a list of objects")

    nodes: List[TextNode] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            # Skip non-dict entries but keep going
            print(f"Skipping non-dict entry at index {idx}")
            continue

        # Build a readable text representation for embedding
        parts = []
        for k, v in item.items():
            try:
                part_val = json.dumps(v, ensure_ascii=False)
            except Exception:
                part_val = str(v)
            parts.append(f"{k}: {part_val}")
        text = "\n".join(parts)

        node = TextNode(text=text)
        # Attach metadata so the original dict is preserved in the vector DB
        # Add some provenance keys as well
        meta = {"source": os.path.basename(file_path), "item_index": idx}
        # Merge the item keys but avoid overwriting provenance
        meta.update(item)
        node.metadata = meta

        nodes.append(node)

    # Optionally generate embeddings if an embed_model was provided
    if embed_model is not None:
        for node in nodes:
            node_embedding = embed_model.get_text_embedding(node.get_content())
            node.embedding = node_embedding

    return nodes


def upload_nodes(nodes: List[TextNode], vector_store) -> int:
    """Add the given nodes to the provided vector_store and return number added."""
    if not nodes:
        return 0
    vector_store.add(nodes)
    return len(nodes)


def _parse_args(argv: Optional[List[str]] = None):
    """Very small arg parser: returns (path, dry_run)."""
    if argv is None:
        argv = sys.argv[1:]
    dry_run = False
    path = "./data/faq.json"

    if argv:
        if argv[0] in ("-n", "--dry-run"):
            dry_run = True
        else:
            path = argv[0]
        if len(argv) > 1 and argv[1] in ("-n", "--dry-run"):
            dry_run = True

    return path, dry_run


if __name__ == "__main__":
    path, dry_run = _parse_args()
    print(f"Loading JSON from: {path} (dry_run={dry_run})")

    # 1) Time to import / launch the model and vector store
    t0 = perf_counter()
    # Import here so the time to create the embed_model is measured
    from model_ve_db import vector_store, embed_model
    t1 = perf_counter()
    model_launch_time = t1 - t0

    # 2) Time to build nodes (includes generating embeddings because we pass embed_model)
    t2 = perf_counter()
    nodes = build_nodes_from_json(path, embed_model=embed_model)
    t3 = perf_counter()
    node_creation_time = t3 - t2

    print(f"Built {len(nodes)} nodes from JSON")

    if dry_run:
        print("Dry run requested; not uploading nodes.")
        print(f"Timings (s): model_launch={model_launch_time:.3f}, node_creation={node_creation_time:.3f}")
        sys.exit(0)

    # 3) Time to upload nodes to the vector store
    t4 = perf_counter()
    added = upload_nodes(nodes, vector_store=vector_store)
    t5 = perf_counter()
    upload_time = t5 - t4

    print(f"Uploaded {added} nodes to vector store")
    print(f"Timings (seconds):")
    print(f"  model launch: {model_launch_time:.3f}s")
    print(f"  node creation (+embeddings): {node_creation_time:.3f}s")
    print(f"  upload: {upload_time:.3f}s")
