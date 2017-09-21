/* Python interface using pybind11 */

#include "libsketchmap.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"

namespace py = pybind11;
using fptype = float;

PYBIND11_MODULE(sketchmap, m) {
    m.doc() = "Sketchmap kernels";

    py::class_<ChiOptions<fptype>>(m, "ChiOptions")
            .def(py::init<bool, bool, bool, bool, bool, fptype>(), "Create an evaluator options object",
                 py::arg("use_switch"), py::arg("use_weights"), py::arg("use_gradient"), py::arg("use_hessian"), py::arg("use_mix"), py::arg("imix"))
            .def_readwrite("use_switch", &ChiOptions<fptype>::use_switch)
            .def_readwrite("use_weights", &ChiOptions<fptype>::use_weights)
            .def_readwrite("use_gradient", &ChiOptions<fptype>::use_gradient)
            .def_readwrite("use_hessian", &ChiOptions<fptype>::use_hessian)
            .def_readwrite("use_mix", &ChiOptions<fptype>::use_mix)
            .def_readwrite("imix", &ChiOptions<fptype>::imix);

    py::class_<Grid<fptype>>(m, "Grid")
            .def(py::init<fptype, fptype, fptype, fptype>(), "Create a grid object",
                 py::arg("min"), py::arg("max"), py::arg("step"), py::arg("eval_step"))
            .def_readwrite("min", &Grid<fptype>::min)
            .def_readwrite("max", &Grid<fptype>::max)
            .def_readwrite("step", &Grid<fptype>::step)
            .def_readwrite("eval_step", &Grid<fptype>::eval_step);

    py::class_<StressFunction<fptype>> sf(m, "StressFunction");
    py::enum_<StressFunction<fptype>::Metric>(sf, "Metric")
            .value("Euclidean", StressFunction<fptype>::Metric::Euclidean)
            .value("Spherical", StressFunction<fptype>::Metric::Spherical)
            .value("Periodic", StressFunction<fptype>::Metric::Periodic);
    py::enum_<StressFunction<fptype>::Backend>(sf, "Backend")
            .value("CPU", StressFunction<fptype>::Backend::CPU)
            .value("OpenCL", StressFunction<fptype>::Backend::OpenCL)
            .value("CUDA", StressFunction<fptype>::Backend::CUDA);
    sf.def(py::init<py::array_t<fptype>, py::array_t<fptype>, fptype, fptype, int, int, int, int, StressFunction<fptype>::Metric, Grid<fptype>>())
    .def_property_readonly("jacobian", &StressFunction<fptype>::getJacobian)
            .def_property_readonly("single_jacobian", &StressFunction<fptype>::getSingleJacobian)
            .def_property_readonly("single_hessian", &StressFunction<fptype>::getSingleHessian)
            .def("eval", &StressFunction<fptype>::eval_py, "Evaluate the stress of the given points",
                 py::arg("x"), py::arg("options"))
            .def("grid_search", &StressFunction<fptype>::grid_search_py, "Minimize per point score using brute force",
                 py::arg("x"), py::arg("index"), py::arg("options"))
            .def("device", &StressFunction<fptype>::device, "Set backend and display info")
    .def("eval_single", &StressFunction<fptype>::eval_single_py, "Evaluate the stress of the given point with all other points",
         py::arg("index"), py::arg("xi"), py::arg("x"), py::arg("options"))
    .def("eval_multi", &StressFunction<fptype>::eval_multi_py, "Evaluate the stress of the given points per point",
         py::arg("x"), py::arg("p"), py::arg("options"));
}

#pragma clang diagnostic pop