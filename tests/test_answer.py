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


# --- D-13: 자기모순 기권 문장 제거 ---
from servers.answer import ABSTAIN_PHRASE, strip_contradictory_abstention  # noqa: E402


def test_trailing_abstention_after_real_answer_is_removed():
    ans = (
        "과학고 입학전형의 공정성을 위해 자기주도학습 전형 규정과 절차를 매뉴얼에 명시하고 "
        "사교육 영향평가를 실시해야 합니다.\n\n출처: 결과보고서_52\n\n" + ABSTAIN_PHRASE
    )
    out = strip_contradictory_abstention(ans)
    assert ABSTAIN_PHRASE not in out
    assert "사교육 영향평가" in out and "출처: 결과보고서_52" in out


def test_pure_abstention_is_kept():
    assert strip_contradictory_abstention(ABSTAIN_PHRASE + ".") == ABSTAIN_PHRASE + "."


def test_abstention_with_only_short_filler_is_kept():
    ans = "압구당시\n" + ABSTAIN_PHRASE
    assert strip_contradictory_abstention(ans) == ans


def test_answer_without_phrase_is_unchanged():
    ans = "정수기는 단체급식용, 저장형, 수도직결형으로 나뉩니다."
    assert strip_contradictory_abstention(ans) == ans


def test_same_line_trailing_abstention_is_removed():
    # D-24 실측: 줄바꿈 없이 같은 문단 안에서 실제 답 뒤에 문장만 바꿔 기권 문구를 붙인 경우.
    # 줄 단위 제거로는 문장 전체(실제 답 포함)가 함께 지워져 길이 안전장치가 오작동했었다.
    ans = (
        "이는 교육 접근성과 질 향상에 기여하며, 모든 학생이 공정한 교육 기회를 받을 수 "
        "있도록 하는 데 중점을 둡니다. " + ABSTAIN_PHRASE + "."
    )
    out = strip_contradictory_abstention(ans)
    assert ABSTAIN_PHRASE not in out
    assert "교육 접근성" in out
