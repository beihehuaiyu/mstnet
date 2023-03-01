"""
模型输出结果->csv
"""

import paddle
from easymia.libs import manager

@manager.PLUGINS.add_component
class ComposePlugins(object):
    """
    TBD
    """
    def __init__(self, plugins):
        self.plugins = plugins
        self.enable = False
        if len(self.plugins) > 0: self.enable = True

    def pre(self, fundus_img, oct_list):
        """
        TBD
        """
        for plugin in self.plugins:
            fundus_img, oct_list = plugin.pre(fundus_img, oct_list)
        return fundus_img, oct_list

    def __call__(self, model, fundus_img, oct_list, real_index):
        fundus_img, oct_list = self.pre(fundus_img, oct_list)
        logits = model(fundus_img, oct_list)
        logits = self.step(logits)
        return self.post(logits, real_index)


    def step(self, logits):
        """
        TBD
        """
        for plugin in self.plugins:
            logits = plugin.step(logits)
        return logits

    def post(self, logits, real_index):
        """
        TBD
        """
        for plugin in self.plugins:
            logits = plugin.post(logits, real_index)
        return logits

    def dump(self, cache):
        """
        TBD
        """
        for plugin in self.plugins:
            logits = plugin.dump(cache)
