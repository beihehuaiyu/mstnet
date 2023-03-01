"""
数据变换器抽象类
"""

class AbstractTransform:
    """
    数据变换器抽象类
    变换器需继承本类，并按需实现__clas__, __det__, __seg__方法
    子类不得重写__call__方法
    """
    def __init__(self, mode):
        """
        目前mode仅支持分类、检测、分割
        """
        assert mode in ['clas', 'det', 'seg', 'pretrain'], \
            "Mode in config file must be `train`, `val`, `test` or `pretrain`, but got {}".format(mode)
        self.mode = mode
    
    def __call__(self, *args):
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
            try:
                out = self.__pretrain__(*args)
            except NotImplementedError:
                out = self.__clas__(*args)
            return out

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