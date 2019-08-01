// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_PYOPERATOR_HPP
#define NETKET_PYOPERATOR_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "abstract_operator.hpp"
#include "py_bosonhubbard.hpp"
#include "py_graph_operator.hpp"
#include "py_local_operator.hpp"
#include "py_pauli_strings.hpp"

namespace py = pybind11;

namespace netket {

void AddOperatorModule(py::module m) {
  auto subm = m.def_submodule("operator");

  auto op =
      py::class_<AbstractOperator>(m, "Operator", R"EOF(
      Abstract class for quantum Operators. This class prototypes the methods
      needed by a class satisfying the Operator concept. Users interested in
      implementing new quantum Operators should derive they own class from this
      class
       )EOF")
          .def("get_conn", &AbstractOperator::GetConn, py::arg("v"), R"EOF(
       Member function finding the connected elements of the Operator. Starting
       from a given visible state v, it finds all other visible states v' such
       that the matrix element O(v,v') is different from zero. In general there
       will be several different connected visible units satisfying this
       condition, and they are denoted here v'(k), for k=0,1...N_connected.

       Args:
           v: A constant reference to the visible configuration.

       )EOF")
          .def_property_readonly(
              "hilbert", &AbstractOperator::GetHilbert,
              R"EOF(netket.hilbert.Hilbert: ``Hilbert`` space of operator.)EOF")
          .def("to_sparse", &AbstractOperator::ToSparse,
               R"EOF(
         Returns the sparse matrix representation of the operator. Note that, in general,
         the size of the matrix is exponential in the number of quantum
         numbers, and this operation should thus only be performed for
         low-dimensional Hilbert spaces or sufficiently sparse operators.

         This method requires an indexable Hilbert space.
         )EOF")
          .def("to_dense", &AbstractOperator::ToDense,
               R"EOF(
         Returns the dense matrix representation of the operator. Note that, in general,
         the size of the matrix is exponential in the number of quantum
         numbers, and this operation should thus only be performed for
         low-dimensional Hilbert spaces.

         This method requires an indexable Hilbert space.
         )EOF")
          .def("to_linear_operator",
               [](py::object py_self) {
                 const auto* cxx_self = py_self.cast<AbstractOperator const*>();
                 const auto dtype =
                     py::module::import("numpy").attr("complex128");
                 const auto linear_operator =
                     py::module::import("scipy.sparse.linalg")
                         .attr("LinearOperator");
                 const auto dim = cxx_self->Dimension();
                 return linear_operator(
                     py::arg{"shape"} = std::make_tuple(dim, dim),
                     py::arg{"matvec"} = py::cpp_function(
                         // TODO: Does this copy data?
                         [py_self, cxx_self](const Eigen::VectorXcd& x) {
                           return cxx_self->Apply(x);
                         }),
                     py::arg{"dtype"} = dtype);
               },
               R"EOF(
        Converts `Operator` to `scipy.sparse.linalg.LinearOperator`.

        This method requires an indexable Hilbert space.
          )EOF")
          .def("__call__", &AbstractOperator::Apply,
               R"EOF(
        Applies the operator to a state.
            )EOF");

  AddBoseHubbard(subm);
  AddLocalOperator(subm);
  AddGraphOperator(subm);
  AddPauliStrings(subm);

  subm.def(
      "local_values",
      [](AbstractOperator& op, AbstractMachine& machine, py::array_t<double, py::array::c_style> samples,
         py::array_t<Complex, py::array::c_style> log_values, Index batch_size) {
        switch (log_values.ndim()) {
          case 2: {
            NETKET_CHECK(samples.ndim() == 3, InvalidInputError,
                         "samples has wrong dimension: " << samples.ndim()
                                                         << "; expected 3.");
            NETKET_CHECK(samples.shape(1) == log_values.shape(1),
                         InvalidInputError, "incompatible number of chains");
            auto local_values = py::cast(LocalValues(
                Eigen::Map<const RowMatrix<double>>{
                    samples.data(), samples.shape(0) * samples.shape(1),
                    samples.shape(2)},
                Eigen::Map<const VectorXcd>{
                    log_values.data(),
                    log_values.shape(0) * log_values.shape(1)},
                machine, op, batch_size));
            local_values.attr("resize")(log_values.shape(0),
                                        log_values.shape(1));
            return local_values;
          }
          case 1:
            NETKET_CHECK(samples.ndim() == 2, InvalidInputError,
                         "samples has wrong dimension: " << samples.ndim()
                                                         << "; expected 2.");
            return py::cast(LocalValues(
                Eigen::Map<const RowMatrix<double>>{
                    samples.data(), samples.shape(0), samples.shape(1)},
                Eigen::Map<const VectorXcd>{log_values.data(),
                                            log_values.shape(0)},
                machine, op, batch_size));
          default:
            NETKET_CHECK(false, InvalidInputError,
                         "log_values has wrong dimension: "
                             << log_values.ndim()
                             << "; expected either 1 or 2.");
        }
      },
      py::arg{"samples"}.noconvert(), py::arg{"log_values"}.noconvert(),
      py::arg{"machine"}, py::arg{"op"}, py::arg{"batch_size"} = 16,
      R"EOF(Computes local values of the operator `op` for all `samples`.

            Args:
                samples: A matrix (or a rank-3 tensor) of visible
                    configurations. If it is a matrix, each row of the matrix
                    must correspond to a visible configuration.  `samples` is a
                    rank-3 tensor, its shape should be `(N, M, #visible)` where
                    `N` is the number of samples, `M` is the number of Markov
                    Chains, and `#visible` is the number of visible units.
                log_values: Corresponding values of the logarithm of the
                    wavefunction. If `samples` is a `(N, #visible)` matrix, then
                    `log_values` should be a vector of `N` complex numbers. If
                    `samples` is a rank-3 tensor, then the shape of `log_values`
                    should be `(N, M)`.
                machine: Wavefunction.
                op: Hermitian operator.
                batch_size: Batch size.

            Returns:
                A numpy array of local values of the operator.)EOF");
}

}  // namespace netket

#endif
