from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader
import torch.distributed as dist


DATALOADERS = {BertDataloader.code(): BertDataloader, AEDataloader.code(): AEDataloader}


def dataloader_factory(args, distributed, rank, world_size):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset, distributed, rank, world_size)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
