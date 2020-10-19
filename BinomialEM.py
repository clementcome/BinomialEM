import numpy as np
from scipy.special import binom
from tqdm import tqdm


class BinomialEM:
    """
    Performing Expectation Maximization algorithm on Binomial data
    """

    def __init__(
        self,
        n_components: int = 1,
        n_estimation: int = 100,
        max_iter: int = 100,
        hide_pbar: bool = False,
    ):
        self.n_components_ = n_components
        self.n_estimation_ = n_estimation
        self.max_iter_ = max_iter
        self.hide_pbar_ = hide_pbar
        self.lambd_ = None
        self.p_ = None

    def f(self, i: int):
        return (
            binom(self.n_estimation_, i)
            * self.p_ ** i
            * (1 - self.p_) ** (self.n_estimation_ - i)
        )

    def fit(self, S: np.ndarray, initial_p: np.ndarray = None):

        self.lambd_ = np.repeat(1 / self.n_components_, self.n_components_)
        if initial_p is not None:
            self.p_ = initial_p
        else:
            min_p = np.min(S / self.n_estimation_)
            max_p = np.max(S / self.n_estimation_)
            self.p_ = np.linspace(min_p, max_p, self.n_components_)
        S = S.reshape((-1, 1))
        n_sample = S.shape[1]

        for t in tqdm(range(self.max_iter_), disable=self.hide_pbar_):
            # E-step
            P = np.apply_along_axis(self.f, 1, S)

            # M-step
            self.lambd_ = np.sum(P, axis=0) / n_sample
            self.p_ = np.sum(P * S, axis=0) / (
                n_sample * self.lambd_ * self.n_estimation_
            )

    def predict(self, S: np.ndarray):
        S = S.reshape((-1, 1))
        P = np.apply_along_axis(self.f, 1, S)
        labels = np.apply_along_axis(np.argmax, 1, P)
        return labels
