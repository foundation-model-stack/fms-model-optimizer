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
"""FMS Model Optimizer init. Import most commonly used functions and classes here."""

# Standard
from importlib.metadata import PackageNotFoundError, version
import logging

# Local
try:
    from fms_mo.prep import qmodel_prep
    from fms_mo.utils.qconfig_utils import qconfig_init
except ImportError:
    pass


VERSION_FALLBACK = "0.0.0"

try:
    __version__ = version("fms_mo")
except PackageNotFoundError:
    __version__ = VERSION_FALLBACK
