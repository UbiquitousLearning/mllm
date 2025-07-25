# RotLLM  


RotLLM adds **rotation-based quantization** support to the MLLM framework.  

With a few commands you can export both the rotated FP32 model and the rotated-and-quantized QNN (HTP) model.

### 1. Install the Rotation Package
```bash
cd mllm/RotLLM
pip install -e .
```

### 2. Export Your Models
Follow the instructions in [README](./mllm_qnn_convertor/README.md) to:

* Export the **rotated FP32 model**  
* Export the **rotated & quantized QNN model**

That’s it—your rotated and quantized MLLM is ready to deploy!