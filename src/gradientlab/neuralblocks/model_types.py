import torch
from typing import Dict, List, Optional, Tuple, TypeAlias


Block_KVCache: TypeAlias = Optional[Dict[str, torch.Tensor]]
Model_KVCache: TypeAlias = Optional[List[Block_KVCache]]

AttnMask: TypeAlias = Optional[torch.Tensor]

Dataset_VQA_Batch: TypeAlias = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
