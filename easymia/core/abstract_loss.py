"""
损失函数抽象类
"""

import paddle

class AbstractLoss(paddle.nn.Layer):
    """
    损失函数抽象类
    其它损失函数需继承本类，并按需实现__clas__, __det__, __seg__方法
    子类不得重写__call__方法
    """
    def __init__(self, mode):
        """
        目前mode仅支持分类、检测、分割、预训练
        """
        assert mode in ['clas', 'det', 'seg', 'pretrain'], \
            "Mode in config file must be `train`, `val`, `test` or `pretrain`, but got {}".format(mode)
        self.mode = mode
        super(AbstractLoss, self).__init__()
    
    def forward(self, *args):
        """
        根据mode调用相应的func
        """
        if self.mode == "clas":
            return self.__clas__(*args)
        elif self.mode == "det":
            return self.__det__(*args)
        elif self.mode == "seg":
            return self.__seg__(*args)
        elif self.mode == "pretrain":
            return self.__pretrain__(*args)

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