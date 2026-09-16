"""답변 생성 보조 함수 단위 테스트 (VLM 서버 없이 실행 가능)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from servers.answer import _text_char_budget, normalize_for_vlm  # noqa: E402
from config import settings  # noqa: E402


def test_compatibility_ideograph_is_normalized():
    # D-12: U+F9CA(CJK 호환 한자 '流')가 Ollama qwen2.5vl 러너의 퇴행 출력을 유발했다.
    text = "유입수(流入水) 중에 들어있는 오염물질"
    out = normalize_for_vlm(text)
    assert "流" not in out
    assert "流" in out  # 일반 한자 流
    assert out == "유입수(流入水) 중에 들어있는 오염물질"


def test_normal_korean_text_is_unchanged():
    text = "학교에서 다친 경우 급여청구를 부모가 할 수 있습니다."
    assert normalize_for_vlm(text) == text


def test_budget_shrinks_with_images():
    b0, b1, b2 = _text_char_budget(0), _text_char_budget(1), _text_char_budget(2)
    assert b0 > b1 > b2 >= 300
    expected0 = int(
        (settings.vlm_num_ctx - settings.answer_max_tokens - settings.answer_context_reserved_tokens)
        * settings.answer_context_chars_per_token
    )
    assert b0 == expected0
