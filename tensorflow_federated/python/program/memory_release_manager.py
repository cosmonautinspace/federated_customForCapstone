# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to release values from a federated program to memory."""

import collections
from typing import Any, Dict, Hashable

from tensorflow_federated.python.program import release_manager


class MemoryReleaseManager(release_manager.ReleaseManager):
  """A `ReleaseManager` that stores values in memory."""

  def __init__(self):
    """Returns an initialized `MemoryReleaseManager`."""
    self._values = collections.OrderedDict()

  def release(self, value: Any, key: Hashable):
    """Releases a value from a federated program.

    Args:
      value: The value to release.
      key: A hashable value to use to reference the released `value`.
    """
    self._values[key] = value

  def values(self) -> Dict[Hashable, Any]:
    """Returns all values keyed by the reference used to release the value."""
    return self._values.copy()
