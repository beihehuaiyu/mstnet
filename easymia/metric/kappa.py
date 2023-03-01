"""
ACC metric
"""

import numpy as np
import paddle
import paddle.nn.functional as F
from easymia.libs import manager
from sklearn.metrics import cohen_kappa_score

@manager.METRICS.add_component
class KappaMetric(object):
    """
    Negative Log Loss for metric
    return 1 - nll(prob, label)
    """
    def __init__(self, weight='quadratic'):
        """
        Init
        """
        self.name = "kappa"
        self.y1 = []
        self.y2 = []
        self.y3 = []
        self.weight = weight

    def step(self, logits, labels):
        """
        step
        logits: paddle tensor, shape with [N, C, d1, d2, ...]
        labels: paddle tensor, shape with [N, C, d1, d2, ...]
        """
        assert paddle.sum(logits, axis = -1).shape == labels.shape
        labels = labels.numpy()
        logits1 = logits.numpy()
        logits2 = logits.numpy().argmax(1)

        self.y1.extend(labels)
        self.y2.extend(logits1)
        self.y3.extend(logits2)

    def clear(self):
        """
        clear
        """
        self.y1 = []
        self.y2 = []
        self.y3 = []
    
    def calc(self):
        labels = np.array(self.y1)
        pred1 = np.array(self.y2)
        pred2 = np.array(self.y3)
        avg_kappa = cohen_kappa_score(pred2, labels, weights=self.weight)
        cross_entropy = F.cross_entropy(paddle.to_tensor(pred1), paddle.to_tensor(labels))
        ret_info = "kappa = {:.4f} , cross_entropy_loss = {:.6}".format(avg_kappa, cross_entropy.item())
        return ret_info


    @property
    def benchmark(self):
        """
        benchmark
        """
        labels = np.array(self.y1)
        pred = np.array(self.y3)
        avg_kappa = cohen_kappa_score(pred, labels, weights=self.weight)
        return avg_kappa