"""Phase 0 검증 스크립트.

검증 기준: 샘플 이미지 1장에 대해 한국어 설명이 돌아오면 통과.
D-02 검증 기준 추가: 페이지당 캡셔닝 소요 시간을 측정해
7B가 이 하드웨어(RTX 4070 Laptop 8GB)에서 실사용 가능한 속도인지 확인한다.
"""

from __future__ import annotations

import io
import sys
import time

from PIL import Image, ImageDraw, ImageFont

from servers.vlm_client import VLMClient


def make_sample_image() -> Image.Image:
    """텍스트/표 형태가 섞인 샘플 이미지를 만든다 (외부 파일 의존 없이 재현 가능하게)."""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "2026년 3분기 매출 현황", fill="black")
    rows = [
        ("구분", "매출액(억원)", "전기대비"),
        ("반도체", "1,240", "+8.2%"),
        ("가전", "560", "-2.1%"),
        ("모바일", "890", "+3.4%"),
    ]
    y = 80
    for row in rows:
        x = 20
        for cell in row:
            draw.text((x, y), cell, fill="black")
            x += 220
        y += 40
    return img


def main() -> int:
    client = VLMClient()

    print(f"VLM 서버 확인: {client._client.base_url} (model={client.model})")
    if not client.ping():
        print("실패: VLM 서버에 연결할 수 없습니다. `ollama serve` 및 모델 pull 상태를 확인하세요.")
        return 1
    print("서버 응답 확인됨.")

    image = make_sample_image()
    prompt = (
        "이 문서 페이지의 내용을 한국어로 설명하라. 표는 행·열 구조를 유지해 "
        "마크다운으로, 차트는 축·수치·추세를 구체적으로 기술하라."
    )

    t0 = time.time()
    caption = client.caption(image, prompt)
    elapsed = time.time() - t0

    print(f"\n--- 캡션 (소요 {elapsed:.1f}초) ---")
    print(caption)

    has_korean = any("가" <= ch <= "힣" for ch in caption)
    if not caption.strip():
        print("\n실패: 빈 응답.")
        return 1
    if not has_korean:
        print("\n실패: 한국어 응답이 아닙니다.")
        return 1

    print("\n통과: 한국어 캡션 생성 확인.")
    if elapsed > 40:
        print(f"경고: 페이지당 {elapsed:.1f}초는 D-02의 40초 상한을 초과합니다. 3B 재검토 필요.")
    elif elapsed > 20:
        print(f"주의: 페이지당 {elapsed:.1f}초. 예상(20초)보다 느립니다 — GPU 상주 여부를 확인하세요.")
    else:
        print(f"양호: 페이지당 {elapsed:.1f}초, 예상 범위 내.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
