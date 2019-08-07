// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_MACHINE_PY_ABSTRACT_MACHINE_HPP
#define NETKET_MACHINE_PY_ABSTRACT_MACHINE_HPP

#include "Machine/abstract_machine.hpp"

namespace netket {

class PyAbstractMachine : public AbstractMachine {
 public:
  PyAbstractMachine(std::shared_ptr<const AbstractHilbert> hilbert)
      : AbstractMachine{std::move(hilbert)} {}

  int Nvisible() const override { return GetHilbert().Size(); }
  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;
  Complex LogValSingle(VisibleConstType v, const any & /*unused*/) override;
  any InitLookup(VisibleConstType /*unused*/) override;
  void UpdateLookup(VisibleConstType /*unused*/,
                    const std::vector<int> & /*unused*/,
                    const std::vector<double> & /*unused*/,
                    any & /*unused*/) override;
  VectorType DerLogSingle(VisibleConstType v, const any & /*lt*/) override;
  ~PyAbstractMachine() override = default;

  /// Functions which one needs to override from Python
  int Npar() const override;
  bool IsHolomorphic() const noexcept override;
  void LogVal(Eigen::Ref<const RowMatrix<double>> v, Eigen::Ref<VectorXcd> out,
              const any & /*unused*/) override;
  void DerLog(Eigen::Ref<const RowMatrix<double>> v,
              Eigen::Ref<RowMatrix<Complex>> out, const any &cache) override;
  void Save(const std::string &filename) const override;
  void Load(const std::string &filename) override;
  PyObject *StateDict() override;
};

}  // namespace netket

#endif  // NETKET_MACHINE_PY_ABSTRACT_MACHINE_HPP
