"""
分类数据集基类
"""
import paddle

class Dataset(paddle.io.Dataset):
    """
    数据集基类
    """
    def __init__(self, split):
        """
        Init
        """
        split = split.lower()
        assert split in ['train', 'val', 'test', 'pretrain'], \
            "Arg split in config file must be `train`, `val`, `test` or `pretrain`, but got {}".format(split)
        self.split = split

    @staticmethod
    def collate_fn(batch):
        """
        default paddle.fluid.dataloader.collate.default_collate_fn
        """
        return paddle.fluid.dataloader.collate.default_collate_fn(batch)