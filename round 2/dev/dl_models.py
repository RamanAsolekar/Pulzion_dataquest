import sys
from huggingface_hub import snapshot_download
models = sys.argv[1:]
for m in models:
    try:
        p = snapshot_download(m, allow_patterns=["*.json","*.txt","*.model","*.safetensors","sentencepiece*","*.py","tokenizer*","modules.json","1_Pooling/*"], ignore_patterns=["*onnx*","*openvino*","*.bin","*.h5","*.msgpack","*.ot"])
        print("OK", m, p, flush=True)
    except Exception as e:
        print("FAIL", m, repr(e)[:200], flush=True)
