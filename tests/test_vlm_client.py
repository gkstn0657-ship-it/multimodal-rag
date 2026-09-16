"""VLM 클라이언트의 퇴행 출력 감지 (D-12) 단위 테스트. 서버 없이 실행 가능."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from servers.vlm_client import looks_degenerate  # noqa: E402


def test_repeated_at_signs_is_degenerate():
    assert looks_degenerate("@" * 31)


def test_two_char_alphabet_is_degenerate():
    assert looks_degenerate("ababababababababab")


def test_normal_korean_answer_is_not_degenerate():
    text = "학교에서 다친 경우 급여청구를 부모가 할 수 있습니다. 절차는 다음과 같습니다."
    assert not looks_degenerate(text)


def test_short_output_is_not_flagged():
    # 짧은 정상 답("부록", "예")을 퇴행으로 오판하지 않는다
    assert not looks_degenerate("부록")
    assert not looks_degenerate("예.")


def test_mostly_one_char_with_noise_is_degenerate():
    assert looks_degenerate("@" * 40 + "a b")
