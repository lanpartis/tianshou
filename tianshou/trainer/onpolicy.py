import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional
import pandas as pd
from logging import Logger

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def onpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        step_per_epoch: int,
        collect_per_step: int,
        repeat_per_collect: int,
        episode_per_test: Union[int, List[int]],
        batch_size: int,
        pretrain_fn: Optional[Callable[[BasePolicy, int], None]] = None,
        prelearn_fn: Optional[Callable[[BasePolicy, int], None]] = None,
        pretest_fn: Optional[Callable[[BasePolicy, int], None]] = None,
        postepoch_fn: Optional[Callable[[int, float], None]] = None,
        stop_fn: Optional[Callable[[int, dict], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy, dict], None]] = None,
        log_fn: Optional[Callable[[dict], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1,
        verbose: bool = True,
        test_in_train: bool = True,
        logger: Logger = None,
        start_epoch: int = 1,
        result_df: pd.DataFrame = pd.DataFrame()
) -> Dict[str, Union[float, str]]:
    """A wrapper for on-policy trainer procedure. The ``step`` in trainer means
    a policy network update.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of episodes the collector would
        collect before the network update. In other words, collect some
        episodes and do one policy network update.
    :param int repeat_per_collect: the number of repeat time for policy
        learning, for example, set it to 2 means the policy needs to learn each
        given batch data twice.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :type episode_per_test: int or list of ints
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function train_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of training in this
        epoch.
    :param function test_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of testing in this
        epoch.
    :param function save_fn: a function for saving policy when the undiscounted
        average mean reward in evaluation phase gets better.
    :param function stop_fn: a function receives the average undiscounted
        returns of the testing result, return a boolean which indicates whether
        reaching the goal.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.
    :param bool test_in_train: whether to test in the training phase.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    global_step = start_epoch * collect_per_step
    best_epoch, best_reward = -1, -1.
    stat = {}
    start_time = time.time()
    test_in_train = test_in_train and train_collector.policy == policy
    for epoch in range(start_epoch, 1 + max_epoch):
        # train
        policy.train()
        if pretrain_fn:
            pretrain_fn(policy, epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:
                result = train_collector.collect(n_episode=collect_per_step)
                data = {}
                if test_in_train and stop_fn and stop_fn(epoch, result, best_reward):
                    test_result = test_episode(
                        policy, test_collector, pretest_fn,
                        epoch, episode_per_test, writer, global_step)
                    if stop_fn and stop_fn(epoch, result, best_reward):
                        if save_fn:
                            save_fn(policy, test_result, best_reward, epoch)
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result['rew'], df=result_df)
                    else:
                        policy.train()
                        if pretrain_fn:
                            pretrain_fn(policy, epoch)
                if prelearn_fn:
                    prelearn_fn(policy, epoch)
                losses = policy.update(
                    0, train_collector.buffer, batch_size, repeat_per_collect)
                train_collector.reset_buffer()
                step = 1
                for k in losses.keys():
                    if isinstance(losses[k], list):
                        step = max(step, len(losses[k]))
                global_step += step * collect_per_step
                for k in result.keys():
                    data[k] = f'{result[k]:.2f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            'train/' + k, result[k], global_step=global_step)
                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f'{stat[k].get():.6f}'
                    if writer and global_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=global_step)
                data_df = pd.DataFrame(data, index=[0])
                result_df = result_df.append(data_df, ignore_index=True)
                t.update(step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(policy, test_collector, pretest_fn, epoch,
                              episode_per_test, writer, global_step)
        if postepoch_fn:
            postepoch_fn(epoch=epoch, reward=result["rew"],result_df=result_df)
        if save_fn:
            save_fn(policy, result, best_reward, epoch)
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
        if verbose:
            pt = print
            if logger:
                pt = logger.info
            pt(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
               f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if stop_fn and stop_fn(epoch, result, best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward, df=result_df)