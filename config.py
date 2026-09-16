"""프로젝트 전역 설정.

VLM/임베더/리랭커/저장소 경로를 한 곳에서 관리한다.
D-02 결정에 따라 VLM 컨텍스트는 4k로 제한하고,
인덱싱 단계와 서빙 단계에서 임베더 device를 다르게 쓴다
(인덱싱: cuda, 서빙: cpu — VLM과 GPU를 동시에 점유하지 않기 위함).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- VLM (Ollama, OpenAI 호환 엔드포인트) ---
    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_model: str = "qwen2.5vl:7b"
    vlm_api_key: str = "ollama"
    vlm_num_ctx: int = 4096  # D-02: 8k -> 4k
    vlm_timeout_sec: float = 120.0

    # --- 임베딩 / 리랭킹 ---
    embed_model_name: str = "BAAI/bge-m3"
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    embed_device_indexing: str = "cuda"  # 인덱싱 시점: VLM을 내린 뒤 GPU 사용
    embed_device_serving: str = "cpu"  # 서빙 시점: VLM이 GPU를 점유하므로 CPU

    # BGE-M3 기본 max_seq_length=8192. 코퍼스 텍스트는 p99=2,747자(~700토큰)라
    # 배치 안에 긴 텍스트 하나만 있어도 전체 배치가 그 길이까지 패딩되어
    # 실측 GPU 100%인데도 배치(64건)당 7~8초로 나왔다(D-05). 1024토큰으로 제한.
    embed_max_seq_length: int = 1024
    embed_batch_size: int = 128

    # --- 검색 파라미터 ---
    top_k_candidates: int = 10
    top_n_after_rerank: int = 5
    answer_image_cap: int = 2  # D-02: 4k 컨텍스트 제약으로 원본 이미지 최대 2장만 투입
    answer_context_max_chars_per_chunk: int = 500  # D-09: 청크 텍스트 5개 + 이미지 2장 합산이 4k 초과(실측 5,519토큰)
    answer_max_tokens: int = 500

    # --- 라우팅 (Phase 1, D-04 4갈래) ---
    route_min_text_chars: int = 100  # 이상이면 텍스트 경로
    route_min_ocr_chars: int = 100  # 텍스트 빈약 + OCR 이상이면 OCR 경로 (VLM 생략)
    blank_ink_threshold: float = 0.002  # 400px 썸네일에서 어두운 픽셀 비율 미만이면 백지
    image_route_page_cap: int = 5000  # D-03: 이미지(VLM) 경로 상한

    # --- 캡셔닝 (D-04) ---
    caption_parallel: int = 4  # 실측: 4요청에서 장당 3.7초, 6요청 3.5초로 포화
    caption_max_tokens: int = 400  # 549토큰 outlier(17.6초) 방지
    caption_cache_path: Path = Path(__file__).resolve().parent / "data" / "captions.jsonl"

    # --- 저장소 ---
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    vector_store_dir: Path = project_root / "data" / "vector_store"  # D-06: 로컬 브루트포스 벡터 저장소
    corpus_parquet: Path = project_root / "data" / "sds_kopub_vdr" / "SDS-KoPub-corpus.parquet"
    qa_parquet: Path = project_root / "data" / "sds_kopub_vdr" / "SDS-KoPub-QA.parquet"
    annotations_parquet: Path = project_root / "data" / "sds_kopub_vdr" / "SDS-KoPub-annotations.parquet"
    rendered_pages_dir: Path = project_root / "data" / "rendered_pages"


settings = Settings()
