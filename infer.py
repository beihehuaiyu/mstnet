import argparse
import random

import paddle
import numpy as np

from easymia.libs import manager, Config
from easymia.utils import get_sys_env, logger
from easymia.trainer import Predict


def parse_args():
    """
    command args
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--infer_splition',
        dest='infer_splition',
        help='Which splition to inference',
        type=str,
        choices=['train', 'val', 'test'],
        default="val")
    parser.add_argument(
        '--multigpu_infer',
        dest='multigpu_infer',
        action='store_true')

    return parser.parse_args()


def main(args):
    """
    main
    """
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        batch_size=args.batch_size,
        resume_model=args.resume_model)

    if args.infer_splition == "train":
        infer_dataset = cfg.train_dataset
    elif args.infer_splition == "val":
        infer_dataset = cfg.val_dataset
    elif args.infer_splition == "test":
        infer_dataset = cfg.test_dataset
    else:
        raise RuntimeError(
            'The eval dataset can\'t be acquired \
                from the configuration file when `infer_splition` is {}.'.format(args.infer_splition))

    if infer_dataset is None:
        raise RuntimeError(
            'The eval dataset is not specified in the configuration file.')
    elif len(infer_dataset) == 0:
        raise ValueError(
            'The length of infer_dataset is 0. Please check if your dataset and splition is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    runner = Predict(model=cfg.model, 
                      infer_task_loader=infer_dataset, 
                      batch_size=cfg.batch_size,
                      num_workers=args.num_workers,
                      infer_plugins=cfg.infer_plugin,
                      multigpu_infer=args.multigpu_infer)

    runner.infer()


if __name__ == '__main__':
    args = parse_args()
    main(args)