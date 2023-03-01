from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import CosineAnnealingDecay

from easymia.libs import manager

@manager.SCHEDULES.add_component
class WarmupCosine(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self, warmed_lr, warmup_steps, decay_steps, **kwargs):
        start_lr = 0.0
        lr_sch = CosineAnnealingDecay(warmed_lr, decay_steps)

        super(WarmupCosine, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            end_lr=warmed_lr)

        self.update_specified = False