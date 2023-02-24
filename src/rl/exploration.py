from abc import ABC, abstractmethod


class EpsGreedyStrategy(ABC):

    def __init__(self):
        self.step = 0

    @abstractmethod
    def get_next_eps(self) -> float:
        pass

    @abstractmethod
    def get_current_eps(self) -> float:
        pass


class ConstantEpsGreedyStrategy(EpsGreedyStrategy):

    def __init__(self, eps: float):
        super(ConstantEpsGreedyStrategy, self).__init__()
        self.eps = eps

    def get_next_eps(self) -> float:
        return self.eps

    def get_current_eps(self) -> float:
        return self.eps


class ExpDecayEpsGreedyStrategy(EpsGreedyStrategy):

    def __init__(self, start_eps: float, end_eps: float, steps: int):
        super(ExpDecayEpsGreedyStrategy, self).__init__()
        self.current_eps = start_eps
        self.decay_factor = (end_eps / start_eps) ** (1 / steps)
        self.end_eps = end_eps
        self.steps = steps

    def get_next_eps(self) -> float:
        if self.step >= self.steps:
            return self.end_eps
        self.current_eps = self.current_eps if self.step == 0 else self.current_eps * self.decay_factor
        self.step += 1
        return self.current_eps

    def get_current_eps(self) -> float:
        return self.current_eps


class LinDecayEpsGreedyStrategy(EpsGreedyStrategy):

    def __init__(self, start_eps: float, end_eps: float, steps: int):
        super(LinDecayEpsGreedyStrategy, self).__init__()
        self.current_eps = start_eps
        self.end_eps = end_eps
        self.step_eps_decay = (end_eps - start_eps) / steps
        self.steps = steps

    def get_next_eps(self) -> float:
        if self.step >= self.steps:
            return self.end_eps
        self.current_eps = self.current_eps if self.step == 0 else self.current_eps + self.step_eps_decay
        self.step += 1
        return self.current_eps

    def get_current_eps(self) -> float:
        return self.current_eps
