import torch
import numpy as np
import random
import os
from util import get_logger


def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Safe distributed setup:
    # 1) never call get_world_size before init_process_group
    # 2) initialize only when launched by torchrun (WORLD_SIZE > 1)
    # 3) fallback to single process world_size=1
    if not hasattr(args, 'local_rank') or args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available():
        if world_size_env > 1 and not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)

        if torch.distributed.is_initialized():
            args.world_size = torch.distributed.get_world_size()
        else:
            args.world_size = 1
    else:
        args.world_size = 1

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args, logger


def init_device(args, local_rank, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu
