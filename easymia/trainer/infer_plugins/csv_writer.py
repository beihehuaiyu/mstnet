"""
模型输出结果->csv
"""

from datetime import datetime
import pandas as pd
import paddle.nn.functional as F
from easymia.libs import manager

@manager.PLUGINS.add_component
class CSVWritePlugin(object):
    """
    模型输出结果->csv
    """
    def __init__(self, logit_idx=0, save_path=None, only_infer=True):
        """
        use_argmax: bool,    是否需要执行argmax操作
        argmax_dim: int,     对哪个维度执行argmax操作
        logit_idx:  int,     模型输出为一个list，要记录list中第几个元素的输出
        save_path:  None|str,csv记录的路径
        """
        self.logit_idx = logit_idx
        self.only_infer = only_infer

        if save_path is None:
            self.save_path = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + "_csvwriter.csv"
        else:
            self.save_path = save_path

        self.file_handle = None

    def pre(self, fundus_img, oct_list):
        """
        TBD
        """
        return fundus_img, oct_list

    def step(self, logits):
        """
        TBD
        """
        return logits

    def post(self, logits, real_index):
        """
        TBD
        """
        logits = logits[0].numpy().argmax()
        real_index = real_index[0]
        return logits, real_index

    def dump(self,cache):
        """
        step
        """
        submission_result = pd.DataFrame(cache, columns=['data', 'dense_pred'])
        submission_result = pd.DataFrame(cache, columns=['data', 'dense_pred'])
        submission_result['non'] = submission_result['dense_pred'].apply(lambda x: int(x == 0))
        submission_result['early'] = submission_result['dense_pred'].apply(lambda x: int(x == 1))
        submission_result['mid_advanced'] = submission_result['dense_pred'].apply(lambda x: int(x == 2))
        submission_result[['data', 'non', 'early', 'mid_advanced']].to_csv("./submission_sub1.csv", index=False)
