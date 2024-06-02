## Football DashBoard



Setup VirtualEnv:
```
virtualenv -p $(which python3.10) --system-site-packages football_env
source football_env/bin/activate
pip install -r requirements.txt
```


## Memory Gotachas

Deallocate fragmented memory:
```
import torch
torch.cuda.empty_cache() 
```

Environment tweaks:
```
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```
