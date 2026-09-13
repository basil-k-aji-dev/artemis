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

"""Unit tests for UnifiedMobileController.open_url shell quoting."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from artemis.controllers.unified_controller import UnifiedMobileController
from artemis.drivers.mock.mock_driver import MockDeviceDriver


@pytest.fixture
def controller(monkeypatch):
    """A controller whose driver records the command string it is handed."""
    ctx = MagicMock()
    ctx.device = MagicMock()
    ctx.device.device_id = "emulator-5554"
    ctx.device.mobile_platform = "android"

    driver = MockDeviceDriver(device_id="emulator-5554")
    driver.execute_shell = AsyncMock(return_value="")
    monkeypatch.setattr(
        "artemis.controllers.unified_controller.get_driver",
        lambda _ctx: driver,
    )
    return UnifiedMobileController(ctx)


@pytest.mark.asyncio
async def test_open_url_quotes_a_url_that_would_break_out_of_single_quotes(controller):
    """A URL containing a single quote must not start a new shell command.

    Wrapping the value in literal single quotes is not enough: the embedded
    quote closes the wrapper and everything after it is parsed by the device's
    own shell.
    """
    await controller.open_url("http://x'; reboot; echo '")

    command = controller.driver.execute_shell.call_args.args[0]
    assert command == (
        """am start -a android.intent.action.VIEW -d 'http://x'"'"'; reboot; echo '"'"''"""
    )
    # Everything after the scheme stays inside one quoted word, so `reboot`
    # is an argument to `am`, never a command of its own.
    assert not command.endswith("reboot; echo ''")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://example.com/path?a=1&b=2",
        "market://details?id=com.android.chrome",
    ],
)
async def test_open_url_passes_ordinary_urls_as_one_word(controller, url):
    """Ordinary URLs still reach `am start` unchanged, as a single argument."""
    import shlex

    assert await controller.open_url(url) is True

    command = controller.driver.execute_shell.call_args.args[0]
    assert command == f"am start -a android.intent.action.VIEW -d {shlex.quote(url)}"
    # Round-tripping through the shell parser yields the original URL.
    assert shlex.split(command)[-1] == url
