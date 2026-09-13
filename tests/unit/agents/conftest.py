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

"""Shared fixtures for agent unit tests."""

from typing import Any

import pytest


class StubChatModel:
    """Stand-in for a resolved provider client.

    Mirrors the only attribute ``VisualStepSummarizer.__init__`` reads back
    (``model``), so the constructor's model-name resolution is still
    exercised, and refuses to be invoked: a test that needs generation must
    install its own fake on ``summarizer._llm``.
    """

    def __init__(self, model_name: str) -> None:
        self.model = model_name

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "The stubbed summarizer model was invoked. A unit test that needs "
            "generation must assign its own fake to `summarizer._llm`."
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<StubChatModel model={self.model!r}>"


@pytest.fixture
def stub_summarizer_model_factories(monkeypatch):
    """Resolve the summarizer's model without provider credentials.

    ``VisualStepSummarizer.__init__`` asks ``get_llm`` for the utility model
    and falls back to ``get_google_llm`` on *any* exception, so with no
    ``GOOGLE_API_KEY`` present both paths end in a real
    ``ChatGoogleGenerativeAI`` construction that raises. Replacing
    ``summarizer._llm`` after the fact is too late -- the object never gets
    built. Both factories are therefore stubbed at the summarizer module
    boundary, which leaves the real summarizer, its retry loop, the
    compressor, the ledger and the runner under test.
    """

    def _fake_get_google_llm(model_name: str | None = None, **_kwargs: Any) -> StubChatModel:
        return StubChatModel(model_name or "gemini-2.5-flash-lite")

    def _fake_get_llm(_ctx: Any, *, name: str = "", is_utils: bool = False, **_kwargs: Any):
        return StubChatModel("gemini-2.5-flash-lite")

    monkeypatch.setattr(
        "artemis.agents.flash.summarizer.get_google_llm",
        _fake_get_google_llm,
    )
    monkeypatch.setattr("artemis.agents.flash.summarizer.get_llm", _fake_get_llm)
