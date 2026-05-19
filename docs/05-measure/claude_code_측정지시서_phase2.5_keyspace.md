# Claude Code 측정 지시서 — Phase 2.5: recall 키 공간 분해능 판정

> **출처**: 컨테이너 실측 6회. 커밋 `3d2d611` + Phase 1·2 (xfail 고정) 완료 상태 기준.
> **성격**: 이것은 **측정 지시서**다. 코드 수정 지시가 아니다.
>          Phase 3 의 규모("모델 교체(소)" vs "비-임베딩 키 설계(대)")를
>          가르는 단일 측정. 결과가 Phase 3 지시서의 입력이 된다.
> **원칙**: 논증으로 정하지 않는다. 측정값이 정한다 (지난 6회 동일).

---

## 0. 왜 이 측정인가 — 확정된 것과 미확정된 것

### 확정 (측정 1~6, 변경 불가)

| 측정 | 결론 |
|---|---|
| 거짓 양성 | 현재 시스템 100% (threshold 0.6 무력) |
| 처방 4종 | threshold/margin/dist-cut 전부 FAIL — 튜닝 레이어 해법 없음 |
| 발견 A | 지표 불일치(cos/L2) **반증** (corr -0.997) |
| 발견 B | query-prefix 누락 **확인** (부분 개선) |
| **keyspace** | **interpretation-key 도 FAIL.** TP cos 0.820~0.832 / HARD_NEG 0.815~0.845 → 음의 마진. 키 *내용* 문제 아님 = **공간 문제 확정** |

→ 후보 (가) interpretation-key, (나) hybrid 는 **측정으로 폐기**.
   둘 다 e5-small 공간을 쓰므로 같은 벽.
→ Claude Code 자기평가의 "키 한 줄 교체 / 절반 실패" 는 **반증됨**.
   처방은 키 텍스트 교체가 아니라 *키 공간 자체*의 문제.

### 미확정 (이 측정이 가른다)

공간 문제의 원인이 둘 중 무엇인가:

- **(원인 X) 차원 부족** — e5-**small** 384d 가 좁아서. 더 큰 모델이면 분리됨.
  → Phase 3 = **소규모** (모델만 교체).
- **(원인 Y) 임베딩 패러다임 한계** — dense sentence embedding 자체가
  "같은 도메인 다른 메커니즘" 을 못 가름. 모델 키워도 안 됨.
  → Phase 3 = **대규모** (비-임베딩 키 설계: 이산 라벨 / sparse 구조화 키).

**이 측정 하나가 Phase 3 의 작업량을 수십 줄 vs 새 서브시스템으로 가른다.**
추측 금지. 측정만이 가른다.

---

## 1. 측정 과제

### 작업 M1. 측정 스크립트 6종을 repo 에 영구 이관 (선행)

지금까지 컨테이너에서만 돌린 스크립트들이 repo 에 없다. 재현성을
위해 먼저 커밋한다. 별첨으로 전달되는 다음 파일을 `scripts/` 에
**로직 수정 없이 그대로** 커밋:

- `conflict_recall_fp_eval.py` (거짓 양성)
- `conflict_recall_remedy_eval.py` (처방 비교)
- `conflict_recall_keyspace_eval.py` (내용 vs 공간 판정)

> 결과 JSON 도 함께 커밋: `docs/03-analysis/conflict_recall_*_raw.json`.
> 이것이 "왜 (가)(나) 를 폐기했나" 의 영구 근거다. 없으면 다음
> cycle 이 같은 측정을 또 한다.

### 작업 M2. 분해능 모델 스윕 (핵심 측정)

**파일**: `scripts/conflict_recall_keyspace_eval.py` 를 모델 인자
받도록 **최소 확장**. 로직(anchor/probe/판정식)은 **건드리지 말 것** —
재현성 깨짐. 모델만 파라미터화.

`EmbeddingBridge` 는 이미 `model_name` 인자를 받는다 (확인됨:
`EmbeddingBridge.DEFAULT_MODEL = "intfloat/multilingual-e5-small"`,
생성자 `model_name: str = DEFAULT_MODEL`). 따라서:

```python
# main() 진입부에 추가
import sys
model = sys.argv[1] if len(sys.argv) > 1 else EmbeddingBridge.DEFAULT_MODEL
enc = EmbeddingBridge(model_name=model)
```

**측정 대상 모델 3종** (순서대로, 각각 독립 실행):

| # | 모델 | dim | 목적 |
|---|---|---|---|
| 1 | `intfloat/multilingual-e5-small` | 384 | baseline (측정 5 재현 확인) |
| 2 | `intfloat/multilingual-e5-large` | 1024 | 차원 ↑ 효과 (원인 X 검정) |
| 3 | `BAAI/bge-m3` | 1024 | 다른 패러다임/학습 (원인 Y 교차검정) |

> 모델 2·3 은 ~2GB 다운로드. 시간 소요 정상. timeout 넉넉히.
> bge-m3 가 환경에서 안 받아지면 `jhgan/ko-sroberta-multitask`
> (한국어 특화, 가벼움) 로 대체 가능 — 단 dim 다름을 보고에 명시.

### 작업 M3. 판정 (스크립트가 자동 출력, 사람이 재확인)

스크립트의 기존 판정식 그대로:
- `완전분리 = TP.min - HN.max > 0` → 분리 성공
- `평균마진 = TP.mean - HN.mean > 0.02` → 부분 분리
- 그 외 → 분리 실패

**원인 판정 규칙** (M2 세 결과를 종합):

| e5-small | e5-large | bge-m3 | 원인 | Phase 3 |
|---|---|---|---|---|
| FAIL | **분리/부분** | — | **X 차원부족** | 소: 모델 교체 |
| FAIL | FAIL | **분리/부분** | X' 패러다임 의존 | 중: 모델 신중 선택 |
| FAIL | FAIL | FAIL | **Y 패러다임 한계** | 대: 비-임베딩 키 |

> e5-small 이 측정 5(컨테이너) 와 다른 수치를 내면 **먼저 보고**.
> 환경 차이(모델 캐시/버전) 의심 — 그 경우 baseline 부터 재정렬.

---

## 2. 절대 금지 (Phase 1·2 §1 계승 + 추가)

1. ❌ threshold 동적 조정 / recall key 텍스트 교체 — 측정으로 무효.
2. ❌ keyspace 스크립트의 anchor/probe/판정식 수정 — 재현성 파괴.
   모델 파라미터화 외 로직 변경 일체 금지.
3. ❌ 이 측정 결과로 곧장 recall_conflict 코드 수정 — **측정 지시서다.
   처방은 Phase 3 별도 지시서.** 결과만 내고 멈춘다.
4. ❌ xfail 2건(Phase 1·2) GREEN 전환 시도 — 처방 전까지 RED 유지.
5. ❌ 외부 LLM 리뷰로 측정 대체 — 이번 cycle 의 유일한 교훈
   ("PDCA Check 는 측정이지 LLM 리뷰 아니다") 위반.

---

## 3. 완료 보고 형식 (표만, 산문 금지)

```
## Phase 2.5 측정 결과

| 모델 | dim | TP cos (min~max) | HARD_NEG cos (min~max) | 완전분리 | 평균마진 | 판정 |
|------|-----|------------------|------------------------|----------|----------|------|
| e5-small  | 384  | ... | ... | ... | ... | FAIL(재현확인) |
| e5-large  | 1024 | ... | ... | ... | ... | ?    |
| bge-m3    | 1024 | ... | ... | ... | ... | ?    |

원인 판정: X 차원부족 / X' 패러다임의존 / Y 패러다임한계  (택1, §1.M3 표 근거)
Phase 3 규모: 소 / 중 / 대
스크립트 6종 커밋: (경로)
결과 JSON 커밋: (경로)
baseline 재현: 측정 5 와 일치 / 불일치(→사유)
금지항목 §2 위반: 없음 (필수)
```

추가로 `CLAUDE.md` 에 1줄 append:
`Phase 2.5: keyspace 공간문제 원인 = {X/X'/Y}. Phase 3 규모 {소/중/대} 확정.`

---

## 4. 한 줄 요약 (가장 먼저 읽을 것)

> **코드 안 고친다.** keyspace 스크립트에 모델 인자만 추가해 3개
> 모델로 돌리고, TP/HARD_NEG cos 분포가 분리되는 모델이 있는지만
> 본다. 분리되면 Phase 3 = 모델 교체(소). 아무 모델도 못 가르면
> Phase 3 = 비-임베딩 키 설계(대). 처방은 다음 지시서. 측정만 하고 멈춘다.
