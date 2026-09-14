# multimodal-rag

텍스트·스캔·표·차트가 섞인 문서를 대상으로 하는 멀티모달 RAG.
텍스트 레이어가 있는 페이지는 텍스트로, 텍스트가 빈약한 페이지는 VLM 캡셔닝으로
라우팅해 인덱싱하고, 답변 시에는 원본 페이지 이미지를 VLM에 직접 입력한다.

- 환경: RTX 4070 Laptop (8GB), Python 3.12, Windows 11
- VLM: Qwen2.5-VL-7B (Ollama 서빙, OpenAI 호환 API, 컨텍스트 4096)
- 임베딩/리랭킹: BGE-M3, bge-reranker-v2-m3
- 벡터 저장소: ChromaDB
- 코퍼스: [SDS KoPub VDR](https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark) (한국 공공문서 361건, 40,781페이지)

실행 계획은 [docs/plan.md](docs/plan.md), 데이터셋 선정 근거는 [docs/dataset.md](docs/dataset.md),
설계 결정과 트레이드오프는 [docs/decisions.md](docs/decisions.md) 참고.

## 왜 vLLM이 아니라 Ollama인가 / 왜 7B에 컨텍스트 4k인가

계획 초안은 Linux + vLLM + 12GB GPU를 가정했으나 실제 개발 환경은
Windows 11(WSL 미설치) + 8GB GPU다. 이에 따라 서빙 엔진과 컨텍스트 길이를
조정했다. 배경과 대가는 [docs/decisions.md](docs/decisions.md)의 D-01, D-02 참고.

## 구조

```
config.py          전역 설정 (VLM/임베더/검색 파라미터)
servers/            검색·답변·API
  vlm_client.py       Ollama(OpenAI 호환) VLM 클라이언트
  retrieve.py         임베딩 검색 + 재랭킹
  answer.py           검색 결과 -> VLM 답변 생성
  main.py             FastAPI /ask 엔드포인트
  metrics.py          단계별 지연시간 계측
indexing/           코퍼스 로딩·라우팅·캡셔닝·임베딩
  corpus.py           SDS KoPub 파켓 스트리밍 리더
  ingest.py           텍스트/이미지 경로 라우팅
  caption.py          VLM 캡셔닝 프롬프트
  embed_store.py      BGE-M3 임베딩 + ChromaDB 저장
  run_index.py        인덱싱 오케스트레이터 (캡셔닝 -> VLM 언로드 -> GPU 임베딩)
eval/               평가셋·judge·리포트 (Phase 3, 예정)
docs/               계획·데이터셋·결정 기록
scripts/            운영 스크립트 (일일 커밋 등)
```

## 실행 순서

```bash
uv sync

# 1) Ollama에 VLM 모델 준비
ollama pull qwen2.5vl:7b

# 2) Phase 0 검증: VLM 응답 확인
.venv/Scripts/python test_vlm.py

# 3) 인덱싱 (스모크 테스트 -> 전체)
.venv/Scripts/python -m indexing.run_index --limit 500
.venv/Scripts/python -m indexing.run_index

# 4) 서버 실행
.venv/Scripts/uvicorn servers.main:app --port 8000
```

## 현재 진행 상태 (2026-09-14)

- [x] Phase 0: VLM 클라이언트, 검증 스크립트
- [x] Phase 1: 라우팅, 캡셔닝, 임베딩·저장 파이프라인 구현
- [x] Phase 2: 검색·재랭킹·답변 생성·FastAPI 엔드포인트 구현
- [x] Phase 5 일부: 단계별 지연시간 계측, VLM 다운/빈 결과 처리
- [ ] 실측 검증: 7B 페이지당 캡셔닝 시간(D-02 40초 기준), 전체 인덱싱 실행
- [ ] Phase 3: 평가셋·judge·baseline 비교 (미착수)
- [ ] Phase 4: ColPali 비교 (선택, 미착수)
