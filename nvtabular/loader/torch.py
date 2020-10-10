#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import torch.distributed as dist
from torch.utils.dlpack import from_dlpack
import cupy as cp
from nvtabular.ops import _get_embedding_order

from .backend import DataLoader


class TorchAsyncItr(torch.utils.data.IterableDataset, DataLoader):
    """This class, creates batches of, a user defined size, tensor
    represenation of the data supplied. The data input requires an
    NVTabular dataset. Handles spillover to ensure all batches are
    the specified size until the final batch.

    Parameters
    -----------
    dataset : NVTabular dataset
    cats : [str]
        the list of categorical columns in the dataset
    conts : [str]
        the list of continuous columns in the dataset
    labels : [str]
        the list of label columns in the dataset
    batch_size : int
        the size of each batch to supply to the model
    shuffle : bool
        enable/disable shuffling of dataset
    parts_per_chunk : int
        number of partitions from the iterator, an NVTabular Dataset, to concatenate into a "chunk"
    devices : [int]
        list representing all available GPU IDs
    """

    def __init__(
        self,
        dataset,
        cats=None,
        conts=None,
        labels=None,
        batch_size=1,
        shuffle=False,
        parts_per_chunk=1,
        device=None,
        callbacks={},
    ):
        self.dl = DataLoader.__init__(
            self,
            dataset,
            cats,
            conts,
            labels,
            batch_size,
            shuffle,
            parts_per_chunk=parts_per_chunk,
            workflows=None,
            device=device,
            callbacks=callbacks
        )

    
    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True


    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()


    def is_distributed(self) -> bool:
        return self.get_world_size() > 1


    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()


    def get_local_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return int(os.environ['LOCAL_RANK'])


    def is_main_process(self):
        return self.get_rank() == 0

        
    def __iter__(self):
        global_size = self.get_world_size()
        rank = self.get_rank()
        # global_size, rank
        indices = self.indices.tolist()
        # grab correct subset
        offset = int(len(indices) / global_size)
        start = int(rank * offset)
        indices = indices[start: start + offset]
        return DataLoader.__iter__(self, indices=indices)

    def _get_device_ctx(self, dev):
        return torch.cuda.device("cuda:{}".format(dev))

    def _to_tensor(self, gdf, dtype=None):
        if gdf.empty:
            return
        dl_pack = gdf.to_dlpack()
#         tens = from_dlpack(dl_pack).type(dtype)
        tens = from_dlpack(dl_pack)
        tens = tens.type(dtype)
        return tens

    # TODO: do we need casting or can we replace this with
    # parent class version?
    def _create_tensors(self, gdf):
        cont_names = self.cont_names
#         if "part_idx" in gdf.columns:
#             cont_names = cont_names + ["part_idx"]
        gdf_cats, gdf_conts, gdf_label = (
            gdf[_get_embedding_order(self.cat_names)],
            gdf[cont_names],
            gdf[self.label_names],
        )
        del gdf
        cats = self._to_tensor(gdf_cats, torch.long)
        conts = self._to_tensor(gdf_conts, torch.float32)
        label = self._to_tensor(gdf_label, torch.float32)
        del gdf_cats, gdf_conts, gdf_label
        return [cats, conts, label]

    def _create_batch(self, tensor, num_samples):
        if tensor is None:
            return []
        idx = self._get_segment_lengths(num_samples)
        return torch.split(tensor, idx)


class DLDataLoader(torch.utils.data.DataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required, to support the FastAI framework.
    """

    def __len__(self):
        return len(self.dataset)
