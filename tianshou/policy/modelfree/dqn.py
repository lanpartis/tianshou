import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from functools import reduce

class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        check_model_interval: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert (
            0.0 <= discount_factor <= 1.0
        ), "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._cnt = 0
        self.check_model_interval = check_model_interval
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(
        self, buffer: ReplayBuffer, indice: np.ndarray
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self(batch, input="obs_next", eps=0).act
            with torch.no_grad():
                target_q = self(
                    batch, model="model_old", input="obs_next"
                ).logits
            target_q = target_q[np.arange(len(a)), a]
        else:
            with torch.no_grad():
                target_q = self(batch, input="obs_next").logits.max(dim=1)[0]
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        eps: Optional[float] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        q, h = model(obs_, state=state, info=batch.info)
        act: np.ndarray = to_numpy(q.max(dim=1)[1])
        if hasattr(obs, "mask"):
            # some of actions are masked, they cannot be selected
            q_: np.ndarray = to_numpy(q)
            q_[~obs.mask] = -np.inf
            act = q_.argmax(axis=1)
        # add eps to act
        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0.0):
            for i in range(len(q)):
                if np.random.rand() < eps:
                    q_ = np.random.rand(*q[i].shape)
                    if hasattr(obs, "mask"):
                        q_[~obs.mask[i]] = -np.inf
                    act[i] = q_.argmax()
        return Batch(logits=q, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch, eps=0.0).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns.flatten(), q)
        td = r - q
        loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        loss.backward()
        self.optim.step()
        self._cnt += 1
        res = {
            "loss": loss.item(), 
            "dist/q":q, 
            "epsilon":self.eps,
        }
        if self._cnt % self.check_model_interval==0:
            model_params = reduce(lambda x,y:torch.cat((x.reshape(-1), y.reshape(-1))) ,map(lambda x:x.data,self.model.parameters()))
            model_params_grad = reduce(lambda x,y:torch.cat((x.reshape(-1), y.reshape(-1))) ,map(lambda x:x.grad,self.model.parameters()))
            res["dist/weight"] = model_params.cpu().numpy()
            res["dist/grad"] = model_params_grad.cpu().numpy() 
        return res
