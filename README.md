# multimodal-rag

텍스트·스캔·표·차트가 섞인 PDF 문서를 대상으로 하는 멀티모달 RAG.
텍스트 레이어가 있는 페이지는 텍스트로, 없는 페이지는 VLM 캡셔닝으로 라우팅해 인덱싱하고,
답변 시에는 원본 페이지 이미지를 VLM에 직접 입력한다.

- 환경: RTX 4070 (12GB), Python 3.11+
- VLM: Qwen2.5-VL-7B-Instruct-AWQ (vLLM 서빙)
- 임베딩/리랭킹: BGE-M3, bge-reranker-v2-m3 (CPU)
- 벡터 저장소: ChromaDB

실행 계획과 단계별 검증 기준은 [docs/plan.md](docs/plan.md) 참고.

## 구조

```
servers/    검색·답변·API
indexing/   PDF 인제스트·캡셔닝·임베딩
eval/       평가셋·judge·리포트
docs/       계획·비용 분석
scripts/    운영 스크립트
```
