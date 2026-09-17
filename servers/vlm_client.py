"""Ollama 네이티브 API(/api/chat) 위에서 Qwen2.5-VL을 호출하는 얇은 클라이언트.

D-01: vLLM 대신 Ollama를 서빙 엔진으로 쓴다.

D-17(2026-09-16): 처음엔 OpenAI 호환 엔드포인트(/v1)를 썼지만, 이 Ollama 버전(0.34.0)의
호환 레이어는 `extra_body={"options": {...}}`로 보낸 num_ctx·num_gpu를 조용히 무시한다
(curl로 직접 확인: /v1/chat/completions에 options를 얹어도 `ollama ps`의 CONTEXT·PROCESSOR가
그대로였다). 같은 옵션을 네이티브 `/api/chat`에 보내면 즉시 반영된다(29/29 레이어 GPU,
CONTEXT 변경 확인). 그래서 OpenAI SDK 대신 requests로 네이티브 엔드포인트를 직접 호출한다.
"""

from __future__ import annotations

import base64
import io
import logging
import subprocess
from pathlib import Path

import requests
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

# D-02 보완: 코퍼스 페이지 이미지는 약 300DPI(2480x3505)로, 원본 그대로 넣으면
# 비전 토큰이 4k 컨텍스트를 넘는다(실측 4,148 토큰 요청 -> 400 에러).
# 긴 변을 이 값으로 제한해 토큰 수를 예측 가능한 범위로 낮춘다.
MAX_IMAGE_SIDE_PX = 1280


def _resize_for_vlm(image: Image.Image) -> Image.Image:
    """긴 변을 MAX_IMAGE_SIDE_PX 이하로 축소한다 (비전 토큰 예산 제어)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_SIDE_PX:
        return image
    scale = MAX_IMAGE_SIDE_PX / longest
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def looks_degenerate(text: str) -> bool:
    """퇴행 출력 감지 (D-12). 같은 글자가 반복되거나 글자 종류가 극단적으로 적으면 True.

    실측: Ollama의 qwen2.5vl 러너가 손상되면 어떤 입력에도 정확히 31개의 '@'만 출력했고,
    `ollama stop` 후 재로드하면 정상으로 돌아왔다. 텍스트 전용 모델은 영향이 없었다.
    """
    s = text.strip()
    if len(s) < 8:
        return False
    if len(set(s)) <= 2:
        return True
    top_char_ratio = max(s.count(c) for c in set(s)) / len(s)
    return top_char_ratio > 0.8


def _image_to_b64(image: Image.Image | bytes | str | Path) -> str:
    """PIL 이미지, 바이트, 또는 파일 경로를 base64 문자열로 변환한다 (data: 접두어 없음, 네이티브 API 형식)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = _resize_for_vlm(image)
        buf = io.BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buf, format="PNG")
        raw = buf.getvalue()
    elif isinstance(image, bytes):
        raw = image
    else:
        raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")
    return base64.b64encode(raw).decode("ascii")


def _native_base_url(base_url: str) -> str:
    return base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url


class VLMClient:
    """Qwen2.5-VL 캡셔닝/답변 생성을 위한 최소 클라이언트 (네이티브 /api/chat)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or settings.vlm_model
        self.num_ctx = num_ctx or settings.vlm_num_ctx
        self.timeout = timeout or settings.vlm_timeout_sec
        self._base_url = _native_base_url(base_url or settings.vlm_base_url)

    def caption(
        self, image: Image.Image | bytes | str | Path, prompt: str, max_tokens: int | None = None
    ) -> str:
        """이미지 한 장에 대한 텍스트 응답(캡션)을 생성한다."""
        messages = [{"role": "user", "content": prompt, "images": [_image_to_b64(image)]}]
        # D-17: 캡셔닝은 짧은 컨텍스트로 충분 (D-02)
        return self._chat(messages, self._load_options(settings.caption_num_ctx), max_tokens)

    @staticmethod
    def _load_options(num_ctx: int) -> dict:
        options = {"num_ctx": num_ctx}
        if settings.vlm_num_gpu is not None:
            options["num_gpu"] = settings.vlm_num_gpu
        return options

    def restart_model(self) -> None:
        """Ollama에 모델 언로드를 요청한다. 다음 요청에서 새로 로드된다 (D-12 회복 절차)."""
        try:
            subprocess.run(["ollama", "stop", self.model], check=False, capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _chat(self, messages: list[dict], options: dict, max_tokens: int | None) -> str:
        """채팅 호출 + 퇴행 출력 감지·회복 (D-12).

        퇴행이 감지되면 모델을 재시작하고 한 번 더 시도한다. 두 번째도 퇴행이면
        빈 문자열이 아니라 예외를 던져 호출자(API는 503)가 알 수 있게 한다.
        """
        payload_options = dict(options)
        if max_tokens:
            payload_options["num_predict"] = max_tokens
        for attempt in range(2):
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, "options": payload_options},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content") or ""
            # D-12 실측: 퇴행 응답은 항상 prompt_eval_count == 0 이었다(정상 응답은 수천).
            # 모델이 프롬프트를 처리하지 않고 반환한 서버 측 실패의 확실한 신호라 함께 검사한다.
            prompt_tokens = data.get("prompt_eval_count")
            not_processed = prompt_tokens == 0
            if not looks_degenerate(content) and not not_processed:
                return content
            logger.warning(
                "VLM 퇴행/미처리 응답 감지 (시도 %d/2, prompt_eval_count=%s, 길이 %d, 앞 20자 %r). 모델을 재시작합니다.",
                attempt + 1, prompt_tokens, len(content), content[:20],
            )
            self.restart_model()
        raise RuntimeError("VLM 러너가 재시작 후에도 퇴행 출력을 반환합니다. Ollama 상태를 확인하세요.")

    def answer_with_images(
        self,
        system_prompt: str,
        user_text: str,
        images: list[Image.Image | bytes | str | Path],
        max_tokens: int | None = None,
    ) -> str:
        """여러 이미지 + 텍스트 컨텍스트로 답변을 생성한다 (답변 상한: settings.answer_image_cap)."""
        capped = images[: settings.answer_image_cap]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text, "images": [_image_to_b64(img) for img in capped]},
        ]
        # D-11: temperature 0으로 답변 생성을 결정적으로 만든다 (평가 재현성)
        return self._chat(
            messages, {**self._load_options(self.num_ctx), "temperature": settings.answer_temperature}, max_tokens
        )

    def ping(self) -> bool:
        """서버가 응답하는지 간단히 확인한다 (이미지 없이)."""
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"num_predict": 5},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception:
            return False
