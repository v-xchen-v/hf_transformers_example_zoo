import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from typing import Union, Optional
import torch
import gc

def _get_dtype(
        dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
    ) -> torch.dtype:
        """Converts `dtype` from `str` to torch.dtype when possible."""
        if dtype is None and config is not None:
            _torch_dtype = config.torch_dtype
        elif isinstance(dtype, str) and dtype != "auto":
            # Convert `str` args torch dtype: `float16` -> `torch.float16`
            _torch_dtype = getattr(torch, dtype)
        else:
            _torch_dtype = dtype
        return _torch_dtype
    
model_name='huggyllama/llama-7b'
print(f'Instantiate model of dtype: default')
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f'Model dtype: {model.dtype}')

del model
gc.collect()
torch.cuda.empty_cache()

# 在 PyTorch 下，模型通常以 torch.float32 格式实例化。如果尝试加载权重为 fp16 的模型，这可能会导致问题，因为它将需要两倍的内存。为了克服此限制，您可以使用 torch_dtype 参数显式传递所需的 dtype：
torch_dtype = _get_dtype('float16')
print(f'Instantiate model of dtype: {torch_dtype}')
model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype)
print(f'Model dtype: {model.dtype}')
del model
gc.collect()
torch.cuda.empty_cache()

# 或者，如果您希望模型始终以最优的内存模式加载，则可以使用特殊值 "auto"，然后 dtype 将自动从模型的权重中推导出：
torch_dtype = _get_dtype('auto')
print(f'Instantiate model of dtype: {torch_dtype}')
model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype)
print(f'Model dtype: {model.dtype}')
del model
gc.collect()
torch.cuda.empty_cache()

# 也可以通过将config中的dtype告知从头开始实例化的模型要使用哪种 dtype：
config = transformers.AutoConfig.from_pretrained(model_name)
torch_dtype=_get_dtype(None, config)
print(f'Instantiate model of dtype: {torch_dtype}')
model = AutoModelForCausalLM.from_pretrained(model_name,  torch_dtype=torch_dtype)
print(f'Model dtype: {model.dtype}')
del model
gc.collect()
torch.cuda.empty_cache()