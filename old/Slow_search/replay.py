import torch
from typing import Callable, List, Optional, Union
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import (EvaluationPlugin,
                                                   default_evaluator)
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.utils import cycle
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from avalanche.benchmarks.utils.utils import concat_datasets


class Replay(SupervisedTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: Optional[int] = None,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        remove_current=False,
        peval_mode="epoch",
        ti=False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
            **kwargs,
        )
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.storage_policy = ClassBalancedBuffer(self.mem_size, adaptive_size=True)
        self.replay_loader = None
        self.remove_current = remove_current
        self.ti = ti

    def _before_training_iteration(self, **kwargs):
        if self.remove_current:
            used = torch.unique(self.mbatch[1]).cpu().tolist()
            buffer = concat_datasets(
                [
                    self.storage_policy.buffer_groups[key].buffer
                    for key, _ in self.storage_policy.buffer_groups.items()
                    if int(key) not in used
                ]
            )
        
            if len(buffer) >= self.batch_size_mem:
                self.replay_loader = cycle(
                    torch.utils.data.DataLoader(
                        buffer,
                        batch_size=self.batch_size_mem,
                        shuffle=True,
                        drop_last=True,
                        num_workers=kwargs.get("num_workers", 0),
                    )
                )

        super()._before_training_iteration(**kwargs)
        
    def _before_training_exp(self, **kwargs):
        if not self.remove_current:
            buffer = self.storage_policy.buffer
        
            if len(buffer) >= self.batch_size_mem:
                self.replay_loader = cycle(
                    torch.utils.data.DataLoader(
                        buffer,
                        batch_size=self.batch_size_mem,
                        shuffle=True,
                        drop_last=True,
                        num_workers=kwargs.get("num_workers", 0),
                    )
                )

        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.replay_loader = None
        #print(str(kwargs))
        self.storage_policy.post_adapt(self, self.experience)#, **kwargs)
        super()._after_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        if self.replay_loader is None:
            return None

        # Augment current task samples
        if not self.ti:
            samples_x, samples_y, _ = next(self.replay_loader)
            samples_x, samples_y = samples_x.to(self.device), samples_y.to(self.device)
            self.mbatch[0] = torch.cat((self.mbatch[0], samples_x), dim=0)
            self.mbatch[1] = torch.cat((self.mbatch[1], samples_y), dim=0)
        else:
            samples_x, samples_y, samples_t = next(self.replay_loader)
            samples_x, samples_y, samples_t = samples_x.to(self.device), samples_y.to(self.device), samples_t.to(self.device)
            self.mbatch[0] = torch.cat((self.mbatch[0], samples_x), dim=0)
            self.mbatch[1] = torch.cat((self.mbatch[1], samples_y), dim=0)
            self.mbatch[2] = torch.cat((self.mbatch[2], samples_t), dim=0)
        

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            if self.ti:
                self.mb_output = self.model(self.mbatch[0], self.mbatch[2])
            else:
                self.mb_output = self.forward()
            self._after_forward(**kwargs)

            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
