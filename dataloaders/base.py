from abc import *
import random
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset, distributed=False, rank=0, world_size=1):
        self.args = args
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size

        # Set random seed for each rank
        # if dist.get_world_size() > 1:
        #     seed = args.dataloader_random_seed + self.rank
        # else:
        seed = args.dataloader_random_seed
        print(f"Setting random seed for dataloader: {seed} (rank: {self.rank})")

        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset["train"]
        self.val = dataset["val"]
        self.test = dataset["test"]
        self.umap = dataset["umap"]
        self.smap = dataset["smap"]
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass

    def _build_sampler(self, dataset, shuffle):
        if self.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
        else:
            sampler = None
        return sampler
