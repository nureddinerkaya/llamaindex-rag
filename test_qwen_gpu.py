"""Quick local test for Qwen3-Embedding GPU readiness.

Usage (on your laptop where CUDA is available):
  1) Check CUDA and driver info:
     python test_qwen_gpu.py --check

  2) (Optional) Attempt to actually instantiate the embedding (will download the model and require enough disk/GPU RAM):
     python test_qwen_gpu.py --load

Notes:
- This script only prints the model kwargs by default. Use --load to try loading the embedding model.
- Run these from the virtualenv/conda environment where you installed the CUDA-enabled torch build.
"""
import argparse


def check_torch():
    try:
        import torch
    except Exception as e:
        print("torch import failed:", e)
        return

    print("torch.__version__:", getattr(torch, "__version__", None))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("CUDA version (torch):", torch.version.cuda)
        except Exception:
            pass
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
        except Exception:
            pass


def show_model_kwargs():
    try:
        import torch
    except Exception:
        torch = None

    cuda_available = (torch is not None) and getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
    if cuda_available:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": getattr(torch, "float16", None),
            "low_cpu_mem_usage": True,
        }
    else:
        model_kwargs = {"device_map": "cpu"}

    print("Computed model_kwargs:")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")


def try_load_embedding():
    # WARNING: This will download and load the model into memory - run only if you know you have space/RAM.
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception as e:
        print("Could not import HuggingFaceEmbedding:", e)
        print("Install llama-index and transformers in your environment before running --load")
        return

    try:
        import torch
    except Exception:
        torch = None

    cuda_available = (torch is not None) and getattr(torch, "cuda", None) is not None and torch.cuda.is_available()
    if cuda_available:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": getattr(torch, "float16", None),
            "low_cpu_mem_usage": True,
        }
    else:
        model_kwargs = {"device_map": "cpu"}

    print("Instantiating HuggingFaceEmbedding (this may take a while)...")
    embed = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs=model_kwargs)
    print("Embed created:", type(embed))
    # Depending on implementation, internals may vary. We don't try to call it here.


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Only check torch/cuda info")
    p.add_argument("--load", action="store_true", help="Attempt to instantiate the HuggingFace embedding (will download model)")
    args = p.parse_args()

    if args.check:
        check_torch()
        show_model_kwargs()
    elif args.load:
        check_torch()
        show_model_kwargs()
        try_load_embedding()
    else:
        p.print_help()

