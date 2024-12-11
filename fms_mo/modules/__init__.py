# Copyright The FMS Model Optimizer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Initialization for modules by import commonly used modules, here so that user can
import from modules directly, instead of import from modules.conv
"""

# First Party
from fms_mo.modules.bmm import QBmm, QBmm_modules
from fms_mo.modules.conv import QConv2d, QConv2d_modules, QConvTranspose2d
from fms_mo.modules.linear import QLinear, QLinear_modules
from fms_mo.modules.lstm import QLSTM, QLSTM_modules
