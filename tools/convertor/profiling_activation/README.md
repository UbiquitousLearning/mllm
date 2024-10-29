## Profiling Activation Tools

### Supported Model Type
- transformers.models.qwen2
- transformers.models.llama
- transformers.models.opt
- transformers.models.gemma
- transformers.models.phi
- transformers.models.mixtral
- transformers.models.falcon

### Examples
1. Get activation distribution config using *get_qwen1.5_act_distribution.py*
```bash
python get_qwen1.5_act_distribution.py
```
**Caution: getting activation distribution config needs huge amount of (cpu) memory. > 100 GB Memory Volume is suggested.**

2. Use activation distribution config to predict in different threshold of clipping.
```bash
python example_run_qwen2_lambada.py
```

### Other Models
1. Modify profiling code
- change model name
- change tokenizer name
- change output file name

2. Modify running code
- change model name
- change config file name