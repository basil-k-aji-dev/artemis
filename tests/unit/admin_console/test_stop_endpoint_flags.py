# Copyright 2026 Google LLC
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

"""`POST /api/stop` must never read a false-like `all` as "stop everything"."""

from typing import Any

from fastapi import HTTPException
import pytest

from apps.admin_console.routers import tasks


class FakeRequest:
    """Minimal stand-in exposing only the `json()` the router awaits."""

    def __init__(self, body: Any, *, raise_value_error: bool = False) -> None:
        self._body = body
        self._raise = raise_value_error

    async def json(self) -> Any:
        if self._raise:
            raise ValueError("no body")
        return self._body


@pytest.fixture
def stop_calls(monkeypatch):
    """Record every `stop_tasks` invocation instead of touching the scheduler."""
    calls: list[dict[str, Any]] = []

    def _stop_tasks(*, clear_all, session_id, device_id):
        calls.append({"clear_all": clear_all, "session_id": session_id, "device_id": device_id})
        return True

    monkeypatch.setattr(tasks.task_queue_service, "stop_tasks", _stop_tasks)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["false", "False", " FALSE ", "no", "off", "0", 0, False])
async def test_false_like_all_does_not_clear_every_task(stop_calls, value):
    """The reported bug: `{"all": "false"}` used to stop every active task."""
    request = FakeRequest({"all": value, "session_id": "session-1"})

    result = await tasks.stop_task(request)

    assert stop_calls == [{"clear_all": False, "session_id": "session-1", "device_id": None}]
    assert result == {"status": "stopped", "session_id": "session-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["true", "True", " ON ", "yes", "1", 1, True])
async def test_true_like_all_still_clears_every_task(stop_calls, value):
    request = FakeRequest({"all": value})

    await tasks.stop_task(request)

    assert stop_calls[0]["clear_all"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["maybe", "", "2", 2, -1, 1.0, [], {}, ["true"]])
async def test_unrecognized_all_is_rejected_before_anything_stops(stop_calls, value):
    request = FakeRequest({"all": value, "session_id": "session-1"})

    with pytest.raises(HTTPException) as caught:
        await tasks.stop_task(request)

    assert caught.value.status_code == 422
    assert "all" in caught.value.detail
    # The decisive assertion: no task was stopped on the way to the error.
    assert stop_calls == []


@pytest.mark.asyncio
async def test_query_all_true_still_wins_over_a_false_body(stop_calls):
    """An explicit `?all=true` keeps precedence, as it did before."""
    request = FakeRequest({"all": "false"})

    await tasks.stop_task(request, all=True)

    assert stop_calls[0]["clear_all"] is True


@pytest.mark.asyncio
async def test_null_all_falls_back_to_the_query_parameter(stop_calls):
    """`{"all": null}` means "not supplied", not "reject" and not "stop all"."""
    await tasks.stop_task(FakeRequest({"all": None, "session_id": "session-1"}))

    assert stop_calls == [{"clear_all": False, "session_id": "session-1", "device_id": None}]


@pytest.mark.asyncio
async def test_non_json_body_still_falls_back_to_query_parameters(stop_calls):
    request = FakeRequest(None, raise_value_error=True)

    await tasks.stop_task(request, all=True, session_id="session-2", device_id="pixel-8")

    assert stop_calls == [{"clear_all": True, "session_id": "session-2", "device_id": "pixel-8"}]
