from typing import Union

from torch import device as TorchDeviceType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch as TorchBackwardPrefetch,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload as TorchCPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy as TorchShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy as TorchModuleWrapPolicy

# General type aliases
PathOrStr = Union[str, "Path"]  # type: ignore

# torch.distributed.fsdp type aliases
FSDPModuleWrapPolicy = TorchModuleWrapPolicy
FSDPShardingStrategy = TorchShardingStrategy
FSDPMixedPrecision = TorchMixedPrecision
FSDPBackwardPrefetch = TorchBackwardPrefetch
FSDPCPUOffload = TorchCPUOffload
Device = TorchDeviceType
