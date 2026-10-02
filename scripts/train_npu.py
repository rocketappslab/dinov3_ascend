"""DINOv3 SSL training entry point for Ascend NPU.

torch_npu's compatibility layer maps .cuda() and NCCL onto NPU and HCCL.
It has to be imported before dinov3.
"""

import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

from dinov3.train.train import main

if __name__ == "__main__":
    main()
