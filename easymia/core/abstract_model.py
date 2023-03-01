"""
模型结构抽象类
"""
import os

import numpy as np
import paddle

from easymia.utils import utils

class AbstractModel(paddle.nn.Layer):
    """
    模型结构抽象类
    其它模型结构需继承本类，并按需实现__clas__, __det__, __seg__方法
    子类不得重写__call__方法
    """
    def __init__(self, mode):
        """
        目前mode仅支持分类、检测、分割、预训练
        """
        assert mode in ['clas', 'det', 'seg', 'pretrain'], \
            "Mode in config file must be `train`, `val`, `test` or `pretrain`, but got {}".format(mode)
        self.mode = mode
        super(AbstractModel, self).__init__()
    
    def forward(self, *args):
        """
        根据mode调用相应的func
        """
        if self.mode == "clas":
            output = self.__clas__(*args)
        elif self.mode == "det":
            output = self.__det__(*args)
        elif self.mode == "seg":
            output = self.__seg__(*args)
        elif self.mode == "pretrain":
            output = self.__pretrain__(*args)

        if isinstance(output, (list, tuple, dict)):
            if isinstance(output, dict):
                depth = utils.dict_depth(output)
                assert depth == 1, \
                    "Output dict must NOT be a recursive dict (depth == 1), got depth = {}".format(depth)
                for key, item in output.items():
                    if isinstance(item, paddle.Tensor):
                        output[key] = [output]
            return output
        elif isinstance(output, paddle.Tensor):
            return [output]
        else:
            raise TypeError("Expect the model output in `list`, `tuple`, `dict` or `Tensor`,\
                 got {}.".format(type(output)))
        
    def __clas__(self):
        """
        分类
        """
        raise NotImplementedError

    def __det__(self):
        """
        检测
        """
        raise NotImplementedError

    def __seg__(self):
        """
        分割
        """
        raise NotImplementedError

    def __pretrain__(self):
        """
        预训练
        """
        raise NotImplementedError