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
"""FMS Model Optimizer version"""

try:
    # Local
    from ._version import (  # pylint: disable=unused-import
        __version__,
        __version_tuple__,
    )
except ImportError as e:
    # Standard
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
