import os
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision

import numpy

cityscapes_dataset_dir = "/home/data/yangrq22/data/cityscapes"

# # 1) Get all samples from Cityscapes dataset in sequence
# dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, task="instance", quality_mode="fine",
#                                usage="train", shuffle=False, num_parallel_workers=1)

# 2) Randomly select 350 samples from Cityscapes dataset
dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, num_samples=350, shuffle=True,
                               num_parallel_workers=1)

# # 3) Get samples from Cityscapes dataset for shard 0 in a 2-way distributed training
# dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, num_shards=2, shard_id=0,
#                                num_parallel_workers=1)