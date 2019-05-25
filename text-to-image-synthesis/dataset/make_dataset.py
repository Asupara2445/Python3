from .mscoco import MSCOCO
from .bird import CUB
from .flower import Oxford_102_flowers


def Dataset(data_name, image_size, gpu_num, batch_size):
    if data_name == "MSCOCO":return MSCOCO(image_size, gpu_num, batch_size)
    elif data_name == "bird":return CUB(image_size, gpu_num, batch_size)
    elif data_name == "flower":return Oxford_102_flowers(image_size, gpu_num, batch_size)