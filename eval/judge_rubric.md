# 생성 품질 채점 루브릭 (사전등록)

> 이 문서는 `eval/gen_eval_answers.json`을 채점하기 **전에** 작성한다.
> 채점 결과를 보고 나서 기준을 바꾸지 않는다 (`docs/plan.md` Phase 3 원칙).

## 대상

15문항 (text 5 · cross 5 · visual 5, SDS KoPub 라벨셋에서 유일한 IMAGE 경로
정답 문항 1개 포함). `eval/gen_eval_sample.json`에 문항, `eval/gen_eval_answers.json`에
두 조건의 답변이 있다.

## 비교 조건

- **ours**: 검색된 청크 중 원본 이미지가 있는 것(IMAGE·OCR 경로, 최대 2장)을 VLM에 직접 투입.
- **baseline**: 원본 이미지를 투입하지 않고 텍스트(캡션 포함)만 사용. 검색 결과는 ours와 동일 —
  차이는 오직 "VLM이 원본 이미지를 직접 보는가"뿐이다.

## 채점자

이 세션의 Claude(Fable 5.1). 답변을 생성한 Qwen2.5-VL-7B와 다른 모델이라 자기 채점 문제를 피한다.
자동 스크립트가 아니라 이 대화 안에서 1회성으로 채점한다(재현 가능한 스크립트가 아님 — 이유는
`docs/worklog.md` "judge가 필요한 생성 품질 평가" 항목 참고: 이 환경에 API 기반 judge가 없음).

## 배점 (100점, docs/plan.md Phase 3 원안 그대로 승계)

| 항목 | 배점 | 기준 |
|---|---|---|
| 정확성 | 40 | gold_answer의 핵심 사실(숫자·명칭·결론)과 일치하는가. 틀린 사실을 말하면 감점 |
| 완전성 | 25 | gold_answer가 다루는 요점을 답변이 빠짐없이 다루는가 |
| 출처정확도 | 20 | 답변 끝의 출처 표기가 실제 근거 문서와 일치하는가. 출처 누락도 감점 |
| 간결성 | 15 | 불필요한 반복·장황함 없이 질문에 맞게 답했는가 |

## 절차

1. 문항마다 gold_answer를 먼저 읽고 핵심 사실을 파악한다.
2. ours와 baseline 답변을 각각 위 배점으로 채점한다(둘 다 gold_answer 대비, 서로 비교하며 채점하지 않는다 — 순서 편향 방지).
3. 점수와 함께 한 줄 근거를 남긴다.
4. 15문항 채점 후 ours/baseline 평균과 유형별(text/cross/visual) 평균을 낸다.
5. 결과를 `eval/generation_report.json`에 저장하고 `docs/worklog.md`에 반영한다.
