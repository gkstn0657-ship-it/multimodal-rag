"""Ollama(OpenAI 호환) 위에서 Qwen2.5-VL을 호출하는 얇은 클라이언트.

D-01: vLLM 대신 Ollama를 서빙 엔진으로 쓴다. OpenAI 호환 엔드포인트를
그대로 쓰므로 이 클라이언트는 서빙 엔진이 바뀌어도 재사용 가능하다.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from openai import OpenAI
from PIL import Image

from config import settings

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


def _image_to_data_url(image: Image.Image | bytes | str | Path) -> str:
    """PIL 이미지, 바이트, 또는 파일 경로를 data URL로 변환한다."""
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
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


class VLMClient:
    """Qwen2.5-VL 캡셔닝/답변 생성을 위한 최소 클라이언트."""

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
        self._client = OpenAI(
            base_url=base_url or settings.vlm_base_url,
            api_key=api_key or settings.vlm_api_key,
            timeout=timeout or settings.vlm_timeout_sec,
        )

    def caption(
        self, image: Image.Image | bytes | str | Path, prompt: str, max_tokens: int | None = None
    ) -> str:
        """이미지 한 장에 대한 텍스트 응답(캡션/답변)을 생성한다."""
        data_url = _image_to_data_url(image)
        kwargs = {"max_tokens": max_tokens} if max_tokens else {}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            # Ollama 확장 파라미터: 컨텍스트 길이 제한 (D-02)
            extra_body={"options": {"num_ctx": self.num_ctx}},
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""

    def answer_with_images(
        self,
        system_prompt: str,
        user_text: str,
        images: list[Image.Image | bytes | str | Path],
        max_tokens: int | None = None,
    ) -> str:
        """여러 이미지 + 텍스트 컨텍스트로 답변을 생성한다 (답변 상한: settings.answer_image_cap)."""
        capped = images[: settings.answer_image_cap]
        content: list[dict] = [{"type": "text", "text": user_text}]
        for img in capped:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

        kwargs = {"max_tokens": max_tokens} if max_tokens else {}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            extra_body={"options": {"num_ctx": self.num_ctx}},
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def ping(self) -> bool:
        """서버가 응답하는지 간단히 확인한다 (이미지 없이)."""
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False
