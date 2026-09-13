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

"""`_resolve_endpoint` must type-check `provider` and `model` like every other field."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from artemis.llm.router import ModelProvider
from artemis.services.llm import _resolve_endpoint


def _ctx(cfg):
    """A context whose agent config is `cfg`."""
    ctx = MagicMock()
    ctx.llm_config.get_agent.return_value = cfg
    ctx.llm_config.get_utils.return_value = cfg
    return ctx


def test_non_string_provider_falls_back_instead_of_raising():
    """The reported bug: a Mock attribute reached `ModelProvider.from_string`.

    `getattr(cfg, "provider", "google")` only returns the default when the
    attribute is *missing*. A config object that has the attribute with a
    non-string value handed it straight through, and `from_string` raised
    "Unknown LLM provider <MagicMock ...>".
    """
    endpoint = _resolve_endpoint(_ctx(MagicMock()), "explorer")

    assert endpoint.provider is ModelProvider.GOOGLE
    assert endpoint.model_name == "gemini-2.5-flash"


@pytest.mark.parametrize("bad", [object(), 123, None, ["google"], {"provider": "google"}])
def test_every_non_string_provider_shape_falls_back(bad):
    endpoint = _resolve_endpoint(_ctx(SimpleNamespace(provider=bad, model=bad)), "explorer")

    assert endpoint.provider is ModelProvider.GOOGLE
    assert endpoint.model_name == "gemini-2.5-flash"


def test_a_real_string_provider_is_still_honoured():
    cfg = SimpleNamespace(provider="openai", model="gpt-4o", temperature=0.7, timeout=30)

    endpoint = _resolve_endpoint(_ctx(cfg), "explorer")

    assert endpoint.provider is ModelProvider.OPENAI
    assert endpoint.model_name == "gpt-4o"
    assert endpoint.temperature == 0.7
    assert endpoint.timeout_seconds == 30


def test_a_provider_enum_value_is_still_honoured():
    """`ModelProvider` is a StrEnum, so the `str` check must not reject it."""
    cfg = SimpleNamespace(provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-5")

    endpoint = _resolve_endpoint(_ctx(cfg), "explorer")

    assert endpoint.provider is ModelProvider.ANTHROPIC
    assert endpoint.model_name == "claude-sonnet-4-5"


def test_a_misspelled_provider_string_still_raises():
    """Type-checking must not turn a real misconfiguration into a silent default.

    Routing a typo to Gemini would use the wrong credentials, which
    `ModelProvider.from_string` deliberately refuses to do.
    """
    cfg = SimpleNamespace(provider="opemai", model="gpt-4o")

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _resolve_endpoint(_ctx(cfg), "explorer")


def test_a_missing_provider_attribute_still_defaults():
    endpoint = _resolve_endpoint(_ctx(SimpleNamespace()), "explorer")

    assert endpoint.provider is ModelProvider.GOOGLE
    assert endpoint.model_name == "gemini-2.5-flash"
