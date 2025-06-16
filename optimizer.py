from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                '''
                TODO: AdamW 구현을 완성하시오. 
                    위의 state 딕셔너리를 사용하여 상태를 읽고 저장하시오.
                    하이퍼파라미터는 `group` 딕셔너리에서 읽을 수 있다(생성자에 저장된 lr, betas, eps, weight_decay).

                        이 구현을 완성하기 위해서 해야할 일들:
                          1. 그래디언트의 1차 모멘트(첫 번째 모멘트)와 2차 모멘트(두 번째 모멘트)를 업데이트.
                          2. Bias correction을 적용(https://arxiv.org/abs/1412.6980 에 제공된 "efficient version" 사용; 프로젝트 설명의 pseudo-code에도 포함됨).
                          3. 파라미터(p.data)를 업데이트.
                          4. 그래디언트 기반의 메인 업데이트 후 weight decay 적용.

                        자세한 내용은 기본 프로젝트 안내문을 참조할 것.
                '''
                ##----- 새로 작성한 코드 -----
                # 상태 초기화 (첫 스텝인 경우)
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Weight decay 적용 (AdamW 방식)
                # 파라미터 업데이트 전에 현재 파라미터 값에 weight decay를 적용
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # 1차 모멘트 (평균) 업데이트
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 2차 모멘트 (제곱 평균) 업데이트
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = alpha / bias_correction1
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    step_size = alpha
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # 파라미터 업데이트
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                ##-------------------------

        return loss
