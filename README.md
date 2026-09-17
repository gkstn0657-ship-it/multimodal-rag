# multimodal-rag

텍스트·스캔·표·차트가 섞인 문서를 대상으로 하는 멀티모달 RAG.
텍스트 레이어가 있는 페이지는 텍스트로, 텍스트가 빈약한 페이지는 VLM 캡셔닝으로
라우팅해 인덱싱하고, 답변 시에는 원본 페이지 이미지를 VLM에 직접 입력한다.
검색은 BGE-M3 밀집 임베딩과 BM25를 RRF로 결합한 하이브리드 방식을 쓴다.

- 환경: RTX 4070 Laptop (8GB), Python 3.12, Windows 11
- VLM: Qwen2.5-VL-7B (Ollama 네이티브 API, 컨텍스트 4096 — 캡셔닝은 3072)
- 임베딩/리랭킹: BGE-M3, bge-reranker-v2-m3, 자체 구현 BM25(RRF 병합)
- 벡터 저장소: numpy 브루트포스(`indexing/vector_store.py`)
- 코퍼스: [SDS KoPub VDR](https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark)
  (한국 공공문서 361건, 40,781페이지) + DART 공시(5개사 사업보고서, 8,418페이지)

성능평가 결과는 [docs/성능평가_보고서.md](docs/성능평가_보고서.md), 데이터셋 선정 근거는
[docs/dataset.md](docs/dataset.md), 설계 결정과 트레이드오프(D-01~D-31)는
[docs/decisions.md](docs/decisions.md), 진행 경과는 [docs/worklog.md](docs/worklog.md) 참고.

## 주요 설계 결정 (요약, 전체는 docs/decisions.md)

- **Ollama, 7B, 컨텍스트 4k**: 계획 초안은 Linux + vLLM + 12GB GPU를 가정했으나 실제 환경은
  Windows 11 + 8GB GPU다. 서빙 엔진과 컨텍스트 길이를 그에 맞춰 조정했다 (D-01, D-02).
- **캡션 프롬프트는 세 번 바뀌었다**: 창작 방지(v2) → 검색 신호가 너무 짧아짐 → 전사 우선(v3).
  캡션·OCR을 합쳐서 임베딩한다 (D-07, D-15).
- **BM25 하이브리드 + 문서명/회사·사업연도 접두어**: 밀집 임베딩만으로는 회사명·연도 같은 고유
  명사 일치가 안 잡혀 DART 질의에서 오답이 잦았다. BM25를 RRF로 얹고 접두어를 더해 개선했다
  (D-19~D-21).
- **기권 게이트는 도입하지 않았다**: 검색 실패 시 무관한 문서로 답하는 문제를 리랭크 점수·점수
  격차·후보 다양성으로 걸러보려 했으나, 세 신호 모두 정답/오답을 깨끗이 못 갈랐다 (D-22).
- **Ollama 네이티브 API 전환**: OpenAI 호환 엔드포인트가 `num_ctx`/`num_gpu` 옵션을 무시해
  GPU 레이어가 일부 CPU로 밀려 있었다. 네이티브 API로 바꿔 해결했다 (D-17).

## 구조

```
config.py          전역 설정 (VLM/임베더/검색/하이브리드 파라미터)
servers/            검색·답변·API
  vlm_client.py       Ollama 네이티브 API VLM 클라이언트 (퇴행 출력 감지·재시작 포함)
  retrieve.py         밀집 검색 + BM25 하이브리드(RRF) + 재랭킹
  answer.py           검색 결과 -> VLM 답변 생성, 기권 문장 후처리
  main.py             FastAPI /ask 엔드포인트
  metrics.py          단계별 지연시간 계측
indexing/           코퍼스 로딩·라우팅·캡셔닝·임베딩
  corpus.py           SDS KoPub 파켓 스트리밍 리더
  pdf_corpus.py       DART 원본 PDF -> 페이지 스트림
  ingest.py           텍스트/이미지 경로 라우팅
  caption.py          VLM 캡셔닝 프롬프트 (v2/v3)
  doc_prefix.py       BM25용 문서명/회사·사업연도 접두어
  bm25_index.py       BM25 인덱스 (자체 구현, 의존성 없음)
  embed_store.py      BGE-M3 임베딩 + 벡터 저장소 저장
  vector_store.py     numpy 브루트포스 벡터 저장소
  run_index.py        SDS 인덱싱 오케스트레이터
  run_index_dart.py   DART 인덱싱 오케스트레이터
  dart_fetch.py       DART 공시 원문 PDF 다운로드
eval/               평가셋·판정·리포트
docs/               계획·데이터셋·결정 기록·성능평가 보고서
scripts/            운영 스크립트 (일일 커밋 등)
```

## 실행 순서

```bash
uv sync

# 1) Ollama에 VLM 모델 준비
ollama pull qwen2.5vl:7b

# 2) 인덱싱 (스모크 테스트 -> 전체)
uv run python -m indexing.run_index --limit 500
uv run python -m indexing.run_index

# 3) BM25 인덱스 (문서명 접두어 포함)
uv run python -m indexing.bm25_index data/vector_store --prefix

# 4) 서버 실행 — http://localhost:8000 에 테스트 화면이 뜬다
#    (질의 → 재랭킹 상위 5페이지의 텍스트·원본 이미지·점수 + 생성 답변을 함께 표시)
uv run uvicorn servers.main:app --port 8000
```

## 평가 재현

```bash
uv run python -m eval.retrieval_eval          # SDS KoPub 600문항 자동 평가
uv run python -m eval.synth_query_eval         # 합성 IMAGE 질의 194건 자동 평가
uv run python -m eval.judge_retrieval_pool     # 판정용 검색 풀 생성
uv run python -m eval.judge_retrieval_score    # 판정 등급 -> 지표 집계
uv run python -m eval.regen_gen_answers        # 생성 품질 재평가용 답변 생성
```

## 현재 상태 (2026-09-17)

- [x] Phase 0~2: VLM 서빙, 라우팅·캡셔닝·임베딩, 검색·답변·API
- [x] Phase 3: SDS 600문항/합성 194건 자동 평가, 42문항 LLM 판정, 생성 품질 재채점(15문항)
- [x] DART 2차 코퍼스 수집·인덱싱, BM25 하이브리드·접두어
- [x] Phase 5: 단계별 지연 계측, 비용 분석([docs/cost_analysis.md](docs/cost_analysis.md))
- [ ] 기권 게이트 (검색 단계 신호로는 미해결, 과제로 보류)
- [ ] ColQwen2 대조 실험 (선택), TEXT 경로 차트 페이지 감지 (선택)
