import sys
import importlib
import importlib.metadata

try:
    import torch
    print("✅ torch imported. version:", torch.__version__)
    print("   PyTorch CUDA version:", torch.version.cuda)
    print("   cuda available:", torch.cuda.is_available())
    print("   torch location:", torch.__file__)
    print("   torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except ImportError as e:
    print("❌ torch import failed:", e)
    sys.exit(1)
try:
    import flash_attn

    version = getattr(flash_attn, "__version__", None)
    if version is None:
        
        version = importlib.metadata.version("flash-attn")
    print("✅ FlashAttention version:", version)
except ImportError:
    print("❌ FlashAttention is not installed.")

try:
    print("==============BEFORE import scgpt==============")
    import scgpt
    print("==============AFTER import scgpt===============")
    print("✅ scgpt imported. version:", scgpt.__version__)
    print("   scgpt location:", scgpt.__file__)
except ImportError as e:
    print("❌ scgpt import failed:", e)
    sys.exit(1)

print("sys.path:", sys.path)


