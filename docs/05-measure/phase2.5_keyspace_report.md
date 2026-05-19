# Phase 2.5 측정 결과 — keyspace 분해능 모델 스윕

**작성일**: 2026-05-20
**지시서**: `docs/05-measure/claude_code_측정지시서_phase2.5_keyspace.md`
**스크립트**: `scripts/conflict_recall_keyspace_eval.py` (모델 인자 파라미터화)
**Raw**: `docs/03-analysis/conflict_recall_keyspace_raw.{e5-small,e5-large,bge-m3}.json`

---

## Phase 2.5 측정 결과

| 모델 | dim | TP cos (min~max) | HARD_NEG cos (min~max) | 완전분리 | 평균마진 | 판정 |
|------|----:|------------------|------------------------|---------:|---------:|------|
| `intfloat/multilingual-e5-small` | 384 | 0.820~0.832 | 0.815~0.845 | -0.0244 | -0.0085 | **FAIL** (재현확인 ✓) |
| `intfloat/multilingual-e5-large` | 1024 | 0.811~0.820 | 0.805~0.834 | -0.0234 | -0.0028 | **FAIL** |
| `BAAI/bge-m3` | 1024 | 0.477~0.495 | 0.415~0.514 | -0.0368 | **+0.0168** | **FAIL** (부분 신호) |

### 추가 부수 측정 (S1 — interpretation 키 상호 분리도)

| 모델 | 다른 충돌 해석 간 cos (min/mean/max) | 분리도 평가 |
|------|--------------------------------------|-------------|
| e5-small | 0.824 / 0.861 / 0.894 | 좁음 (밀집) |
| e5-large | 0.816 / 0.843 / 0.875 | 좁음 (밀집) |
| **bge-m3** | **0.480 / 0.545 / 0.606** | **넓음 (sharper)** |

→ bge-m3 가 *각 해석을 키 공간에서 더 멀리 배치* 함에도 *probe vs 키* 매칭에서는 분리 안 됨.

---

## 원인 판정

지시서 §1.M3 규칙:

| e5-small | e5-large | bge-m3 | 원인 | Phase 3 |
|---------|---------|--------|------|---------|
| FAIL | **분리/부분** | — | X 차원부족 | 소 |
| FAIL | FAIL | **분리/부분** | X' 패러다임 의존 | 중 |
| **FAIL** | **FAIL** | **FAIL** | **Y 패러다임 한계** | **대** |

**3/3 분리 실패 → 원인 Y (패러다임 한계) 확정**

> **Phase 3 규모 = 대.** dense sentence embedding 패러다임 자체가
> "같은 도메인 다른 메커니즘" 을 못 가른다. 모델 키워도 (e5-large 1024d),
> 패러다임 바꿔도 (bge-m3 multi-functionality) 분리 가능선 도달 못 함.
> 후보 (가) interpretation-key, (나) hybrid 폐기에 더해 **(다) 임베딩 모델
> 교체 자체도 단독 해법으로 폐기**. 남은 길: **(라) 비-임베딩 키 — LLM
> 으로 충돌 유형 라벨 추출 → 라벨 임베딩 / 스파스 키 / 구조화 인덱스**.

---

## 부수 관찰 (자동 판정 외, 정성적)

bge-m3 가 *완전 분리* 는 못 했으나 다른 모델과 **질적 차이**:

1. **평균마진만 양수** (+0.0168) — 신호 방향성은 정상 (TP 가 평균적으로 HN 보다 가까움). 단 0.02 미달이라 자동 부분-분리 판정 못 받음.
2. **S1 interpretation 키 상호 분리도 mean 0.545** — e5 (mean ~0.85) 의 절반 수준. **각 해석을 vector space 의 넓은 영역에 분산** 시킴.
3. cos 절대값 자체가 e5 (0.80+) 와 다른 0.45-0.55 영역 — multi-functionality 학습 효과로 보임.

**해석**: bge-m3 가 임베딩 패러다임 안에서는 최선이나, *분리 가능선* 자체가 패러다임 한계.
Phase 3 의 (라) 비-임베딩 키 + bge-m3 의 sharper 분포가 결합 시 hybrid 효과 가능성 — 단 그건 Phase 3+ 의 영역.

---

## 영구 산출물 (M1 + M3)

| 파일 | 경로 | 상태 |
|------|------|------|
| FP 측정 스크립트 | `scripts/conflict_recall_fp_eval.py` | 기 커밋 |
| 처방 비교 스크립트 | `scripts/conflict_recall_remedy_eval.py` | 기 커밋 |
| keyspace 스크립트 (M2 패치) | `scripts/conflict_recall_keyspace_eval.py` | 신규 (sys.argv 추가만) |
| raw JSON e5-small | `docs/03-analysis/conflict_recall_keyspace_raw.e5-small.json` | 신규 |
| raw JSON e5-large | `docs/03-analysis/conflict_recall_keyspace_raw.e5-large.json` | 신규 |
| raw JSON bge-m3 | `docs/03-analysis/conflict_recall_keyspace_raw.bge-m3.json` | 신규 |
| 측정 지시서 | `docs/05-measure/claude_code_측정지시서_phase2.5_keyspace.md` | 기 커밋 |
| 본 리포트 | `docs/05-measure/phase2.5_keyspace_report.md` | 이 문서 |

---

## baseline 재현 확인

지시서 §1.M3 "e5-small 측정 5 와 다른 수치 → 먼저 보고":

| 측정 시점 | TP cos | HARD_NEG cos | 평균마진 |
|----------|--------|--------------|----------|
| 측정 5 (컨테이너, 지시서 §0) | 0.820~0.832 | 0.815~0.845 | (음수, 미공개) |
| 이번 실행 (이 repo) | 0.820~0.832 | 0.815~0.845 | -0.0085 |

→ **완전 일치. 환경 차이 없음. baseline 재현 확인.**

---

## 금지 §2 위반 확인

| 금지 항목 | 위반 여부 |
|----------|:--------:|
| threshold 동적 조정 / recall key 텍스트 교체 | ✓ 없음 |
| anchor/probe/판정식 수정 | ✓ 없음 (sys.argv 만 추가) |
| 측정 결과로 곧장 recall_conflict 코드 수정 | ✓ 없음 (이 cycle 은 측정만) |
| xfail 2건 GREEN 전환 시도 | ✓ 없음 |
| LLM 리뷰로 측정 대체 | ✓ 없음 (오직 수치 판정) |

---

## CLAUDE.md 1줄 append (지시서 §3)

```
Phase 2.5: keyspace 공간문제 원인 = Y (패러다임 한계). Phase 3 규모 = 대 확정.
```

---

## 한 줄 요약

> **3 모델 모두 FAIL — dense sentence embedding 패러다임 자체가 같은 도메인
> 다른 메커니즘을 분리 못 한다. Phase 3 = 대 (비-임베딩 키, LLM 라벨 추출
> 기반). "키 한 줄 교체 / 절반 실패" 자기평가는 다시 반증됨.**
