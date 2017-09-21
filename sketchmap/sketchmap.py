# Ugly workaround until we have proper module names and paths
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sketchmap", "../cmake-build-debug/sketchmap.cpython-35m-darwin.so")
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

from typing import Callable

import numpy as np
from scipy import optimize as opt
from sklearn.manifold import MDS


def minimizable(func):
    def inner(*args, **kwargs):
        arr = args[0]
        npoints = arr.shape[0] // 2
        return func(arr.reshape((npoints, 2)), *args[1:], **kwargs)
    return inner


class SketchMap:
    """
    Sketchmap dimensionality reduction algorithm
    
    """
    def __init__(
            self,
            sigma: float,
            ahigh: int,
            bhigh: int,
            alow: int,
            blow: int,
            metric: str="euclidean",
            nsweeps: int=3,
            mixsteps: int=10,
            maxsteps: int=30,
            tol: float=1e-5
    ):
        self.sigma = sigma
        self.ahigh = ahigh
        self.bhigh = bhigh
        self.alow = alow
        self.blow = blow
        self.nsweeps = nsweeps
        self.mixsteps = mixsteps
        self.maxsteps = maxsteps
        self.tol = tol
        self.history_ = []

        if metric.startswith("euclid"):
            self.metric = sm.StressFunction.Metric.Euclidean
        elif metric.startswith("period"):
            self.metric = sm.StressFunction.Metric.Periodic
        elif metric.startswith("spheric"):
            self.metric = sm.StressFunction.Metric.Spherical
        else:
            e = (
                "{0} is not a valid distance metric! "
                "Valid metrics are: 'euclidean', 'periodic', 'spherical'."
            ).format(metric)
            raise ValueError(e)

    def fit(self, xhigh: np.ndarray, weights: np.ndarray=None):
        self.npoints = xhigh.shape[0]

        if weights is None:
            weights = np.ones(self.npoints, dtype=np.float32)

        assert weights.shape[0] == self.npoints

        # Normalize the weights
        weights /= weights.sum()

        # Create the evaluator
        xhigh = np.asfortranarray(xhigh, dtype=np.float32)
        weights = np.asfortranarray(weights, dtype=np.float32)
        stress = sm.StressFunction(
            xhigh, weights, self.sigma, self.sigma, self.alow,
            self.ahigh, self.blow, self.bhigh, self.metric
        )

        # Create options
        opt_chi2id = sm.ChiOptions(use_switch=False, use_weights=True,
                                   use_gradient=False, use_mix=False, imix=0.0)

        # Initialize with metric dimensional scaling
        xlow = self._mds(xhigh)

        # Start the minimization
        imix = 1.0
        for _ in range(self.mixsteps):
            # Create options
            opt_chi2mix = sm.ChiOptions(use_switch=False, use_weights=True,
                                        use_gradient=True, use_mix=True, imix=imix)

            # New mixing coefficient
            imix = self._new_imix(stress.eval(xlow, opt_chi2mix),
                                  stress.eval(xlow, opt_chi2id))

            # Local minimization
            xlow = self._local_min(stress.eval, xlow, opt_chi2mix)

    def _local_min(
            self,
            func: Callable[[np.ndarray, sm.ChiOptions], float],
            x0: np.ndarray,
            options: sm.ChiOptions
    ) -> np.ndarray:

        return opt.minimize(
            fun=minimizable(func),
            x0=x0.flatten(),
            method="BFGS",
            args=(options,),
            options=dict(maxiter=self.maxsteps),
            tol=self.tol
        ).x.reshape((self.npoints, 2))

    def _mds(self, xhigh: np.ndarray) -> np.ndarray:
        mds = MDS(n_components=2, n_jobs=-1)
        xlow = mds.fit_transform(xhigh)
        return np.asfortranarray(xlow, dtype=np.float32)

    def _new_imix(self, current: float, stress_mix: float, stress_id: float) -> float:
        new = stress_mix / (stress_id + stress_mix)
        new = max(new, 0.5)
        new = min(new, 0.1)
        return new
