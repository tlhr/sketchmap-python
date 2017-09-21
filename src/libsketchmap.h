//
// Created by Thomas on 09/06/17.
//

#ifndef SKETCHMAP_PY_SKETCHMAP_H
#define SKETCHMAP_PY_SKETCHMAP_H

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <pybind11/numpy.h>
#include <limits>
#include "libmath.h"

#define C_INV_TWOPI 0.1591549431
#define C_TWOPI 6.2831853072

namespace py = pybind11;

// Fortran style numpy array
template <typename T> using ndarray =
    py::array_t<T, py::array::f_style | py::array::forcecast>;

/// StressFunction evaluation options. Weights may not be needed
/// in some cases, allowing us to save some multiplications.
/// The gradient is not needed for pointwise global or stochastic minimizers.
template <typename T>
struct ChiOptions {
    ChiOptions(bool use_switch, bool use_weights, bool use_gradient, bool use_mix, T imix):
            use_switch(use_switch), use_weights(use_weights),
            use_gradient(use_gradient), use_mix(use_mix), imix(imix) {}

    bool use_switch, use_weights, use_gradient, use_mix;
    T imix;
};

/// Sigmoid evaluation options. Like with the stress, the gradient may not be required.
/// I am doing this in a struct instead of a simple bool so that the interface is consistent
/// with the `StressFunction` class and to be able to easily extend it in the future.
struct SwitchOptions {
    explicit SwitchOptions(bool use_gradient): use_gradient(use_gradient) {}

    bool use_gradient;
};

/// A struct to conveniently store grid data for pointwise global optimization.
template <typename T>
struct Grid {
    explicit Grid(T min, T max, T npoints, T eval_npoints):
            min(min), max(max), npoints(npoints), eval_npoints(eval_npoints) {
        step = (std::abs(max) + std::abs(min)) / npoints;
        eval_step = (std::abs(max) + std::abs(min)) / eval_npoints;
        center = (max + min) / 2.0;
        range = std::abs(min) + std::abs(max);
    }
    T min, max, step, eval_step, center, range;
    unsigned npoints, eval_npoints;
};

/// Sigmoidal switching function. Using this as a class has the advantage of
/// being able to precalculate some constants, and to take some other computational shortcuts.
template <typename T, typename U>
class Sigmoid {
public:
    Sigmoid() = default;

    Sigmoid(T _sigma, int _a, int _b) : sigma(_sigma), a(_a), b(_b) {
        rsigma = 1.0 / sigma;
        twopowadivbm1 = pow(2.0, static_cast<T>(a) / static_cast<T>(b)) - 1.0;
        negbdivamin1 = -static_cast<T>(b) / static_cast<T>(a) - 1;
        negbdiva = -b / a;
    }

    U eval(U x, SwitchOptions options) {
        const U xrsig = x * rsigma;
        const U mx = twopowadivbm1 * af::pow(xrsig, a);
        if (options.use_gradient) {
            jacobian = b * mx * af::pow(mx + af::constant(1.0, x.dims()), negbdivamin1) / x;
        }
        return 1. - af::pow(1 + mx, static_cast<T>(negbdiva));
    }

    U &getJacobian() {
        return jacobian;
    }

private:
    // Switch parameters
    T sigma;
    int a, b;

    // Derivative
    U jacobian;

    // Precalculate constants
    T rsigma, twopowadivbm1, negbdivamin1;
    int negbdiva;
};

/// Main stress calculation class. We instantiate this in python and can
/// precalculate a large amount of constants.
template <typename T>
class StressFunction {
public:

    enum class Metric {Euclidean, Spherical, Periodic};
    enum class Backend {CPU, OpenCL, CUDA};

    // Internal constructor
    StressFunction(const af::array &_xhigh,
                   const af::array &_weights,
                   Sigmoid<T, af::array> _sigmoid_low,
                   Sigmoid<T, af::array> _sigmoid_high,
                   Metric _distance_metric,
                   Grid<T> _grid);

    // Constructor callable from python
    StressFunction(ndarray<T> _xhigh, ndarray<T> _weights,
                   T _sigma_low, T _sigma_high,
                   int _a_low, int _a_high, int _b_low, int _b_high,
                   Metric _distance_metric,
                   Grid<T> _grid);

    // Scorers
    T eval(af::array &x, ChiOptions<T> options);
    T eval_single(unsigned int index, af::array &xi,
                  const af::array &x, ChiOptions<T> options);
    af::array eval_multi(af::array &x, af::array &p, ChiOptions<T> options);

    ndarray<T> grid_search_py(ndarray<T> _x, unsigned index, ChiOptions<T> options);

    // Pythonized scorers
    T eval_py(ndarray<T> _x, ChiOptions<T> options);
    T eval_single_py(unsigned int index,
                     ndarray<T> _xi,
                     ndarray<T> _x,
                     ChiOptions<T> options);
    ndarray<T> eval_multi_py(ndarray<T> _x, ndarray<T> _p, ChiOptions<T> options);

    ndarray<T> getJacobian() {
        // Copy data from device to host (i.e. the location of the allocated py_array)
        T *jac = jacobian.host<T>();
        ndarray<T> jacpy({static_cast<size_t>(ntotal_low)}, jac);
        return jacpy;
    }

    ndarray<T> getSingleJacobian() {
        // Copy data from device to host (i.e. the location of the allocated py_array)
        T *jac = single_jacobian.host<T>();
        ndarray<T> jacpy({static_cast<size_t>(nldim)}, jac);
        return jacpy;
    }

    std::string device(Backend backend) {
        if (backend == Backend::CUDA) {
            af::setBackend(AF_BACKEND_CUDA);
        } else if (backend == Backend::CPU) {
            af::setBackend(AF_BACKEND_CPU);
        } else if (backend == Backend::OpenCL) {
            af::setBackend(AF_BACKEND_OPENCL);
        }
        return af::infoString();
    }

private:
    long long npoints, nhdim, nldim, ntotal_low;
    Metric distance_metric;

    // Note that checking the correctness of the gradient is not easy using
    // the finite difference method because of the nature of the switching function
    // (i.e. mapping close points "together", or at least closer than the machine epsilon),
    // the ill-conditioned nature of the scoring function, and the use of 32 bit floating point arithmetic.
    af::array jacobian, single_jacobian;

    // Constant data
    af::array xhigh, weights, square_weights, xhighd, fxhighd;
    T fact, rweightsum;

    // Functions
    Sigmoid<T, af::array> sigmoid_low;
    Sigmoid<T, af::array> sigmoid_high;

    // Grid to be used for pointwise global optimization
    Grid<T> grid;

    // Distance
    af::array dist_full(const af::array &x, af::array &diff) const;
    af::array dist_single(const af::array &xi, const af::array &x, af::array &diff) const;
    af::array dist(const af::array &aa, const af::array &bbt, af::array &diff) const;
    af::array dist_multi(const af::array &a, const af::array &b, af::array &diff) const;
};

template <typename T>
StressFunction<T>::StressFunction(const ndarray<T> _xhigh, const ndarray<T> _weights,
                                  T _sigma_low, T _sigma_high,
                                  int _a_low, int _a_high, int _b_low, int _b_high,
                                  Metric _distance_metric,
                                  Grid<T> _grid):
    distance_metric(_distance_metric),
    nldim(2),
    grid(_grid)
{
    npoints = _xhigh.shape(0);
    nhdim = _xhigh.shape(1);
    ntotal_low = npoints * nldim;
    fact = 1.0 / (npoints * (npoints - 1.0));

    // Init switching function options, we don't need the gradient for the
    // high-dimensional distance matrix (it's constant).
    SwitchOptions so(false);

    // Init switching functions
    sigmoid_low = Sigmoid<T, af::array>(_sigma_low, _a_low, _b_low);
    sigmoid_high = Sigmoid<T, af::array>(_sigma_high, _a_high, _b_high);

    // Init constant high dimensional and weight arrays from pointers to flat arrays
    xhigh = af::reorder(af::array(_xhigh.shape(0), _xhigh.shape(1), _xhigh.data(0)), 1, 0);
    weights = af::array(_weights.shape(0), 1, _weights.data(0));

    // Create square weights array, where each element is the product of weight i and j.
    // This makes subsequent operations easier and avoids additional broadcasting / tiling.
    {
        // Broadcast weight arrays into square matrices, one is transposed
        const af::array ww = af::moddims(af::tile(weights, 1, (unsigned)npoints),
                                         1, (unsigned)npoints, (unsigned)npoints);
        const af::array wwt = af::reorder(ww, 0, 2, 1);
        square_weights = ww * wwt;
        rweightsum = 1.0 / af::sum<T>(weights);
    }

    // Precalculate high dimensional distance matrix and switched version
    {
        af::array dummy(nhdim, npoints, npoints);
        xhighd = dist_full(xhigh, dummy);
    }
    fxhighd = sigmoid_high.eval(xhighd, so);
}

// TODO: Probably permanently dead code
template <typename T>
StressFunction<T>::StressFunction(const af::array &_xhigh,
                                  const af::array &_weights,
                                  Sigmoid<T, af::array> _sigmoid_low,
                                  Sigmoid<T, af::array> _sigmoid_high,
                                  Metric _distance_metric,
                                  Grid<T> _grid):
    xhigh(_xhigh),
    weights(_weights),
    sigmoid_low(_sigmoid_low),
    sigmoid_high(_sigmoid_high),
    distance_metric(_distance_metric),
    nldim(2),
    grid(_grid)
{
    npoints = xhigh.dims(0);
    nhdim = xhigh.dims(1);
    ntotal_low = npoints * nldim;
    fact = 1.0 / (npoints * (npoints - 1.0));

    // Init switching function options, we don't need the gradient for the
    // high-dimensional distance matrix (it's constant).
    SwitchOptions so(false);

    // Create square weights array, where each element is the product of weight i and j.
    // This makes subsequent operations easier and avoids additional broadcasting / tiling.
    {
        // Broadcast weight arrays into square matrices, one is transposed
        const af::array ww = af::tile(weights, 1, (unsigned)npoints);
        const af::array wwt = af::reorder(ww, 0, 1);
        square_weights = ww * wwt;
        rweightsum = 1.0 / af::sum<T>(weights);
    }

    // Precalculate high dimensional distance matrix and switched version
    {
        af::array dummy(nhdim, npoints);
        xhighd = dist_full(xhigh, dummy);
    }
    fxhighd = sigmoid_high.eval(xhighd, so);
}

template <typename T>
T StressFunction<T>::eval(af::array &x, ChiOptions<T> options) {
    // Create switching function options
    SwitchOptions so(options.use_gradient);

    // Compute low dimensional distance matrix
    af::array ldiff(nldim, npoints, npoints);
    af::array xlowd = dist_full(x, ldiff);

    // Init intermediate arrays
    af::array lhdiff(npoints, npoints);
    af::array flhdiff(npoints, npoints);
    af::array grad(npoints, npoints);
    af::array fgrad(npoints, npoints);
    T result, fresult = 0.0;

    // Switching function part of the score
    if (options.use_mix || options.use_switch) {
        af::array fxlowd = sigmoid_low.eval(xlowd, so);
        flhdiff = fxhighd - fxlowd;

        if (options.use_gradient) {
            fgrad = flhdiff * sigmoid_low.getJacobian();
        }

        if (options.use_weights) {
            fresult = rweightsum * af::sum<T>(flhdiff * flhdiff * square_weights);
        } else {
            fresult = fact * af::sum<T>(flhdiff * flhdiff);
        }
    }

    // MDS-like part of the score
    if (options.use_mix || !options.use_switch) {
        lhdiff = xhighd - xlowd;

        if (options.use_gradient) {
            grad = lhdiff;
        }

        if (options.use_weights) {
            result = rweightsum * af::sum<T>(lhdiff * lhdiff * square_weights);
        } else {
            result = fact * af::sum<T>(lhdiff * lhdiff);
        }
    }

    // Gradient stuff
    if (options.use_gradient) {
        if (options.use_mix) {
            grad = (1.0 - options.imix) * fgrad + options.imix * grad;
        } else if (options.use_switch) {
            grad = fgrad;
        }
        grad /= xlowd;

        if (options.use_weights) {
            grad *= square_weights;
        }

        jacobian = 4.0 * af::sum(af::tile(
                af::moddims(grad, 1, npoints, npoints), 2) * ldiff, 1, 0.0);

        if (!options.use_weights) {
            jacobian *= fact;
        } else {
            jacobian *= rweightsum;
        }
    }

    // Mix the function
    if (options.use_mix) {
        result = (1.0 - options.imix) * fresult + options.imix * result;
    } else if (options.use_switch) {
        result = fresult;
    }

    return result;
}

template <typename T>
ndarray<T> StressFunction<T>::grid_search_py(ndarray<T> _x,
                                                 unsigned index,
                                                 ChiOptions<T> options) {
    const af::array x = af::reorder(af::array(_x.shape(0), _x.shape(1), _x.data(0)), 1, 0);
    af::array uv = af::constant(0.0, 2);

    struct BestResult {
        T x, y, chi2;
    };

    BestResult res = {0.0, 0.0, std::numeric_limits<float>::max()};

    // Loop through every grid point
    for (T gx = grid.min; gx < grid.max; gx += grid.step) {
        for (T gy = grid.min; gy < grid.max; gy += grid.step) {
            uv(0) = gx;
            uv(1) = gy;
            const T score = eval_single(index, uv, x, options);
            if (score < res.chi2) {
                res.chi2 = score;
                res.x = gx;
                res.y = gy;
            }
        }
    }
    uv(0) = res.x;
    uv(1) = res.y;

    // Copy new array back to host and convert to numpy compatible format.
    T *_uv = uv.host<T>();
    ndarray<T> xi_numpy({static_cast<size_t>(nldim)}, _uv);
    return xi_numpy;
}

/// Calculate many single point stresses in one go. Note that the `p` array is considered weightless,
/// therefore eval multi called with `x` for both inputs will not produce the same result as `eval`.
template <typename T>
af::array StressFunction<T>::eval_multi(af::array &x, af::array &p, ChiOptions<T> options) {
    // Create switching function options
    SwitchOptions so(options.use_gradient);

    // Compute low dimensional distance vector
    af::array ldiff(nldim, npoints);
    af::array xlowd = dist_multi(x, p, ldiff);

    // Init intermediate arrays
    af::array lhdiff(npoints, npoints);
    af::array flhdiff(npoints, npoints);
    af::array grad(npoints, npoints);
    af::array fgrad(npoints, npoints);

    // Output is an array instead of a single number
    af::array result, fresult = af::constant(0.0, npoints);
    const af::array ww = af::moddims(af::tile(weights, 1, (unsigned)npoints),
                                     1, (unsigned)npoints, (unsigned)npoints);

    // Switching function part of the score
    if (options.use_mix || options.use_switch) {
        af::array fxlowd = sigmoid_low.eval(xlowd, so);
        flhdiff = fxhighd - fxlowd;

        if (options.use_gradient) {
            fgrad = flhdiff * sigmoid_low.getJacobian();
        }

        if (options.use_weights) {
            fresult = rweightsum * af::sum(flhdiff * flhdiff * ww, 1, 0.0);
        } else {
            fresult = fact * af::sum(flhdiff * flhdiff, 1, 0.0);
        }
    }

    // MDS-like part of the score
    if (options.use_mix || !options.use_switch) {
        lhdiff = xhighd - xlowd;

        if (options.use_gradient) {
            grad = lhdiff;
        }

        if (options.use_weights) {
            result = rweightsum * af::sum(lhdiff * lhdiff * ww, 1, 0.0);
        } else {
            result = fact * af::sum(lhdiff * lhdiff, 1, 0.0);
        }
    }

    // Gradient stuff
    if (options.use_gradient) {
        if (options.use_mix) {
            grad = (1.0 - options.imix) * fgrad + options.imix * grad;
        } else if (options.use_switch) {
            grad = fgrad;
        }
        grad /= xlowd;

        if (options.use_weights) {
            grad *= ww;
        }

        jacobian = 4.0 * af::sum(af::tile(
                af::moddims(grad, 1, npoints, npoints), 2) * ldiff, 1, 0.0);

        if (!options.use_weights) {
            jacobian *= fact;
        } else {
            jacobian *= rweightsum;
        }
    }

    // Mix the function
    if (options.use_mix) {
        result = (1.0 - options.imix) * fresult + options.imix * result;
    } else if (options.use_switch) {
        result = fresult;
    }

    return result;
}

template <typename T>
T StressFunction<T>::eval_single(unsigned int index, af::array &xi,
                                 const af::array &x, ChiOptions<T> options) {
    // Create switching function options
    SwitchOptions so(options.use_gradient);

    // Compute low dimensional distance vector
    af::array ldiff(nldim, npoints);
    af::array xlowd = dist_single(xi, x, ldiff);

    // Init intermediate arrays
    af::array lhdiff(npoints);
    af::array flhdiff(npoints);
    af::array fgrad = af::constant(0.0, npoints);
    af::array grad = af::constant(0.0, npoints);
    af::array fxlowd(npoints);
    T result, fresult = 0.0;

    // Switching function part of the score
    if (options.use_mix || options.use_switch) {
        fxlowd = sigmoid_low.eval(xlowd, so);
        flhdiff = af::reorder(fxhighd.col(index), 0, 2, 1) - fxlowd;

        if (options.use_gradient) {
            fgrad = flhdiff * sigmoid_low.getJacobian();
        }

        if (options.use_weights) {
            fresult = rweightsum * af::sum<T>(flhdiff * flhdiff * af::reorder(square_weights.col(index), 0, 2, 1));
        } else {
            fresult = fact * af::sum<T>(flhdiff * flhdiff);
        }
    }

    // MDS-like part of the score
    if (options.use_mix || !options.use_switch) {
        lhdiff = af::reorder(xhighd.col(index), 0, 2, 1) - xlowd;

        if (options.use_gradient) {
            grad = lhdiff;
        }

        if (options.use_weights) {
            result = rweightsum * af::sum<T>(lhdiff * lhdiff * af::reorder(square_weights.col(index), 0, 2, 1));
        } else {
            result = fact * af::sum<T>(lhdiff * lhdiff);
        }
    }

    // Gradient stuff
    if (options.use_gradient) {
        if (options.use_mix) {
            grad = (1.0 - options.imix) * fgrad + options.imix * grad;
        } else if (options.use_switch) {
            grad = fgrad;
        }
        grad /= xlowd;

        if (options.use_weights) {
            grad *= af::reorder(square_weights.col(index), 0, 2, 1);
        }

        single_jacobian = -2.0 * af::sum(af::tile(
                af::moddims(grad, 1, npoints), 2) * ldiff, 1, 0.0);

        if (!options.use_weights) {
            single_jacobian *= fact;
        } else {
            single_jacobian *= rweightsum;
        }
    }

    // Mix the function
    if (options.use_mix) {
        result = (1.0 - options.imix) * fresult + options.imix * result;
    } else if (options.use_switch) {
        result = fresult;
    }
    return result;
}

template <typename T>
T StressFunction<T>::eval_single_py(unsigned int index,
                                    const ndarray<T> _xi,
                                    const ndarray<T> _x,
                                    ChiOptions<T> options) {
    af::array x = af::reorder(af::array(_x.shape(0), _x.shape(1), _x.data(0)), 1, 0);
    af::array xi = af::array(_xi.shape(0), _xi.data(0));
    return eval_single(index, xi, x, options);
}

template <typename T>
ndarray<T> StressFunction<T>::eval_multi_py(ndarray<T> _x, ndarray<T> _p, ChiOptions<T> options) {
    af::array x = af::reorder(af::array(_x.shape(0), _x.shape(1), _x.data(0)), 1, 0);
    af::array p = af::reorder(af::array(_p.shape(0), _p.shape(1), _p.data(0)), 1, 0);
    af::array res = eval_multi(x, p, options);
    T *_res = res.host<T>();
    ndarray<T> scores_py({static_cast<size_t>(npoints)}, _res);
    return scores_py;
}

template <typename T>
T StressFunction<T>::eval_py(const ndarray<T> _x,
                             ChiOptions<T> options) {
    af::array x = af::reorder(af::array(_x.shape(0), _x.shape(1), _x.data(0)), 1, 0);
    return eval(x, options);
}

template <typename T>
af::array StressFunction<T>::dist(const af::array &aa, const af::array &bbt, af::array &diff) const {
    // Calculate distance vector (Output is a 3D matrix)
    diff = aa - bbt;

    // Calculate the actual distance matrix using preferred metric
    if (distance_metric == Metric::Euclidean) {
        return af::sqrt(af::sum(diff * diff, 0));
    } else if (distance_metric == Metric::Periodic) {
        af::array dx = (diff) * C_INV_TWOPI;
        dx -= af::round(dx);
        dx *= C_TWOPI;
        return af::sqrt(af::sum(dx * dx, 0));
    } else if (distance_metric == Metric::Spherical) {
        throw std::runtime_error("Spherical distances not implemented yet!");
    } else {
        throw std::runtime_error("Invalid metric specified!");
    }
}

template <typename T>
af::array StressFunction<T>::dist_single(const af::array &xi, const af::array &x, af::array &diff) const {
    const af::array xx = af::tile(xi, 1, (unsigned)npoints);

    return dist(xx, x, diff);
}

template <typename T>
af::array StressFunction<T>::dist_full(const af::array &x, af::array &diff) const {
    // Broadcast arrays into square matrices, one is transposed
    const af::array xx = af::tile(x, 1, 1, (unsigned)npoints);
    const af::array xxt = af::reorder(xx, 0, 2, 1);

    return dist(xx, xxt, diff);
}

template <typename T>
af::array StressFunction<T>::dist_multi(const af::array &a, const af::array &b, af::array &diff) const {
    // Same as dist_full, but operate on two different arrays
    const af::array aa = af::tile(a, 1, 1, (unsigned)npoints);
    const af::array bb = af::tile(b, 1, 1, (unsigned)npoints);
    const af::array bbt = af::reorder(bb, 0, 2, 1);

    return dist(aa, bbt, diff);
}

#endif //SKETCHMAP_PY_SKETCHMAP_H
