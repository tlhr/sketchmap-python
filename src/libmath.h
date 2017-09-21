//
// Created by Thomas on 10/06/17.
//

#ifndef SKETCHMAP_PY_MATH_H
#define SKETCHMAP_PY_MATH_H

#include <cstdlib>
#include <cmath>
#include "arrayfire.h"

#define C_INV_TWOPI 0.1591549431
#define C_TWOPI 6.2831853072

template <typename T>
inline T pow_int(T x, int a) {
    T res = static_cast<T>(1.0);
    for (int i = a; i > 0; --i) {
        res *= x;
    }
    return res;
}

inline af::array pow_int(af::array x, int a) {
    af::array res = af::constant(1.0, x.dims());
    for (int i = a; i > 0; --i) {
        res *= x;
    }
    for (int i = a; i < 0; ++i) {
        res /= x;
    }
    return res;
}

#endif //SKETCHMAP_PY_MATH_H
