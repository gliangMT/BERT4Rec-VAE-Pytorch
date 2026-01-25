import torch
import torch_musa
import torch.distributed as dist
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    local_rank = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    distributed = False
    if world_size > 1:
        print(
            f"Distributed training initialized. Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}"
        )
        distributed = True
    export_root = setup_train(args)

    train_loader, val_loader, test_loader = dataloader_factory(
        args, distributed, rank, world_size
    )
    model = model_factory(args)
    trainer = trainer_factory(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        export_root,
        distributed,
        rank,
    )
    trainer.train()

    test_model = input("Test model with test dataset? y/[n]: ") == "y"
    if test_model:
        trainer.test()


if __name__ == "__main__":
    if args.mode == "train":
        train()
    else:
        raise ValueError("Invalid mode")
