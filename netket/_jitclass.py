# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file permits loading numba's jitclass without throwing deprecation
# warnings in v > 0.49.

# TODO if numba's minimum version is raised to 0.49, remove this file and
# change all usages to from numba.experimental import jitclass

from pkg_resources import get_distribution

if get_distribution("numba").version < "0.49":
    from numba import jitclass
else:
    from numba.experimental import jitclass
