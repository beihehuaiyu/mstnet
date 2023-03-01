"""
训练器
"""

import os
import time
import shutil
from collections import deque

import paddle
import numpy as np

from easymia.utils import utils
from easymia.utils import logger
from easymia.utils import progbar
from .evaluator import Evaluator

import warnings
warnings.filterwarnings("ignore")

class Trainer(Evaluator):
    """
    TBD
    """
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion,
                 epochs,
                 train_task_loader, 
                 val_task_loader, 
                 metrics, 
                 batch_size=[1, 1],
                 num_workers=0,
                 save_dir="./output/",
                 save_interval=100,
                 log_epoch=10,
                 keep_checkpoint_max=5,
                 keep_checkpoint_epoch=None,
                 overwrite_save_dir=False,
                 convert_syncbn=False,
                 infer_plugins=None,
                 multigpu_infer=False):
        """
        TBD
        """
        # super self.model, self.metrics, self.batch_size, self.num_workers, self.val_loader, self.val_samples
        if convert_syncbn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        super().__init__(model, val_task_loader, metrics, batch_size, num_workers, multigpu_infer)
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()

        if nranks > 1:
            paddle.distributed.fleet.init(is_collective=True, strategy=utils.get_strategy())
            self.optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)
            self.ddp_model = paddle.distributed.fleet.distributed_model(model)
        else:
            self.optimizer = optimizer
            self.ddp_model = None

        self.criterion = criterion
        self.epochs = epochs
        self.do_eval = val_task_loader is not None
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_epoch = log_epoch
        self.keep_checkpoint_max = keep_checkpoint_max
        self.keep_checkpoint_epoch = keep_checkpoint_epoch

        self.train_task_loader = train_task_loader
        self.recorder = logger.Recorder(
            ['loss', "reader_cost_time", "batch_cost_time", "lr"] + [m.name for m in metrics])

        self.train_loader, self.train_samples = self.create_loader(self.train_task_loader, self.batch_size[0], True)
        if os.path.exists(self.save_dir) and local_rank == 0:
            if overwrite_save_dir:
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
            else:
                raise ValueError("Output dir {} exists, change output dir \
                    or set `overwrite_save_dir` == True".format(self.save_dir))

    def print_status(self, epoch_id, reset=False):
        """
        print
        """
        avg_loss = self.recorder.get("loss", reduction="mean")

        avg_batch_cost = self.recorder.get("batch_cost_time", reduction="mean")
        avg_reader_cost = self.recorder.get("reader_cost_time", reduction="mean")
        lr = self.recorder.get("lr")[-1]

        logger.info(
            "[TRAIN] epoch: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}"
                    .format(epoch_id, self.epochs, avg_loss, lr, avg_batch_cost, avg_reader_cost))

        if reset:
            self.recorder.clear("loss")
            self.recorder.clear("batch_cost_time")
            self.recorder.clear("reader_cost_time")
            self.recorder.clear("lr")


    def train_epoch(self, epoch_id, iter_epoch):
        """
        模型训练for-loop
        """
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()

        self.model.train()
        batch_start = time.time()

        for iter_id, batch in enumerate(self.train_loader):
            self.recorder.record("reader_cost_time", time.time() - batch_start)
            fundus_img, oct_list, labels = self.unpack_data(batch)
                    
            if nranks > 1:
                outputs = self.ddp_model(fundus_img, oct_list)
            else:
                outputs = self.model(fundus_img, oct_list)
            if isinstance(outputs, dict):
                outputs, labels = self.flatten_dict_output(outputs, labels)
            loss = self.loss_computation(outputs, labels, self.criterion['types'], self.criterion['coef'])
            loss.backward()
            self.optimizer.step()
            self.model.clear_gradients()

            current_lr = self.optimizer.get_lr()
                    # update lr
            if isinstance(self.optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = self.optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = self.optimizer._learning_rate

            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                    lr_sche.step()

            self.recorder.record("batch_cost_time", time.time() - batch_start)
            self.recorder.record("loss", loss.numpy()[0])
            self.recorder.record("lr", current_lr)
                
            batch_start = time.time() 
        if local_rank == 0:
            utils.drop_overtime_files("/dev/shm/", keepSec=3)
        self.model.train()
           

    def fit(self):
        """
        外部调用接口
        """
        local_rank = paddle.distributed.ParallelEnv().local_rank
        iter_epoch = len(self.train_task_loader) + 1
        
        best_benchmark = 0.
        best_model_epoch = 0.
        save_models = deque()

        for epoch_id in range(self.epochs):
            epoch_id = epoch_id+1
            self.train_epoch(epoch_id, iter_epoch)
            if epoch_id % self.log_epoch == 0:
                self.print_status(epoch_id, reset=True)
            if epoch_id % self.log_epoch == 0 or (epoch_id % self.save_interval == 0\
             or epoch_id == self.epochs) and local_rank == 0:
                if self.do_eval:
                    benchmark = self.evaluate()
                    for m in self.metrics:
                        m.clear()
            if (epoch_id % self.save_interval == 0 or epoch_id == self.epochs) and local_rank == 0:   
                current_save_dir = os.path.join(self.save_dir, "epoch_{}".format(epoch_id))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)                
                paddle.save(self.model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(self.optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                if not (self.keep_checkpoint_epoch is not None and epoch_id in self.keep_checkpoint_epoch):
                    save_models.append(current_save_dir)
                if len(save_models) > self.keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    try:
                        shutil.rmtree(model_to_remove)
                    except:
                        logger.info('{} is not exist'.format(model_to_remove))
                if self.do_eval:
                    if benchmark > best_benchmark:
                        best_benchmark = benchmark
                        best_model_epoch = epoch_id

                        best_model_dir = os.path.join(self.save_dir, "best_model")
                        paddle.save(self.model.state_dict(),
                                   os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                            '[EVAL] The model with the best validation benchmark ({:6f}) was saved at epoch {}.'
                            .format(best_benchmark, best_model_epoch))
                 
                