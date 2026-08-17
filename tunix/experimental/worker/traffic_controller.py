# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Admission control for weight sync quiescing."""

import asyncio
import threading


class AdmissionClosedError(RuntimeError):
  """Raised when a request arrives while admission is closed for a sync."""


class TrafficController:
  """Owns the admission gate, sync flag, and in-flight tasks under one lock.

  Admission and sync state always change together, so callers never observe
  a half-transitioned worker.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._admission_open = True
    self._syncing = False
    self._tasks = set()

  def is_admission_open(self) -> bool:
    with self._lock:
      return self._admission_open

  def is_syncing(self) -> bool:
    with self._lock:
      return self._syncing

  def transition_to_syncing(self) -> None:
    """Closes admission and marks the worker as syncing."""
    with self._lock:
      self._admission_open = False
      self._syncing = True

  def reopen(self) -> bool:
    """Reopens admission and clears the sync flag."""
    with self._lock:
      self._admission_open = True
      self._syncing = False
      return True

  def track(self, task) -> None:
    """Registers an in-flight task; it removes itself on completion."""
    with self._lock:
      self._tasks.add(task)
    task.add_done_callback(self._discard)

  def _discard(self, task) -> None:
    with self._lock:
      self._tasks.discard(task)

  async def drain(self, timeout_s: float) -> None:
    """Waits for the tracked tasks to finish, up to the timeout."""
    with self._lock:
      tasks = list(self._tasks)
    if tasks:
      await asyncio.wait(tasks, timeout=timeout_s)
