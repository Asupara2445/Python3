from .mscoco import MSCOCO
from .bird import CUB
from .flower import Oxford_102_flowers


def Dataset(args, pre_train=False):
    if args.name == "MSCOCO":return MSCOCO(args, pre_train)
    elif args.name == "bird":return CUB(args, pre_train)
    elif args.name == "flower":return Oxford_102_flowers(args, pre_train)