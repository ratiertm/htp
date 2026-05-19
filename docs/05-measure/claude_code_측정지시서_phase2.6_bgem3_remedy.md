# Claude Code 측정 지시서 — Phase 2.6: bge-m3 처방 증폭 시험

> **출처**: 컨테이너 실측 6회 + Phase 2.5 (keyspace 모델 스윕) 완료 기준.
> **성격**: 측정 지시서. 코드 수정 아님. Phase 3 규모를
>          **중규모(모델 교체+처방)** vs **대규모(비-임베딩 키 신설)** 로
>          가르는 단일 측정. 결과가 Phase 3 지시서의 입력.
> **원칙**: 측정값이 정한다. Phase 2.5 가 한 칸 건너뛴 빈칸을 메운다.

---

## 0. 왜 이 측정인가 — Phase 2.5 가 건너뛴 빈칸

### Phase 2.5 가 확정한 것 (유효)

3 모델(e5-small/e5-large/bge-m3) raw cos 분포 모두 자동 판정 FAIL.
→ 단일 임베딩 모델 교체만으로는 분리선 도달 못 함. 후보 (가)(나)(다)
폐기.

### Phase 2.5 가 **건너뛴** 것 (이 측정의 대상)

Phase 2.5 리포트는 "3/3 FAIL → 패러다임 한계(Y) 확정 → Phase 3 대"
로 닫았다. 그러나 자동 판정식이 **bge-m3 의 질적 차이를 흡수**했다:

| | e5-small/large | **bge-m3** |
|---|---|---|
| 평균마진 | 음수 (-0.008, -0.003) | **양수 +0.0168** |
| S1 해석 상호분리도 mean | ~0.86 (밀집) | **0.545 (sharper, 절반)** |
| 신호 방향성 | 없음 | **정상 (TP 가 평균적으로 HN 보다 가까움)** |

→ bge-m3 는 e5 와 **질적으로 다른 FAIL**. 신호가 *아예 없는* 것이
아니라 *약하게 존재하나 0.02 자동컷 미달*. 패러다임 한계라면 bge-m3
도 마진 음수여야 하는데 양수다 — **임베딩 안에 신호가 있다는 직접
증거.**

### 측정되지 않은 빈칸

측정 2(remedy: P1 상대마진/P2 분포컷/P3 결합)는 **e5-small 위에서만**
돌렸다. e5-small 은 신호가 0 인 공간 — 증폭기는 0 을 증폭 못 하므로
당시 FAIL 은 당연. **"bge-m3 + 측정 2 처방" 조합은 아무도 측정한 적
없다.** 약한 신호 + 증폭기 = ?

> 이 빈칸을 안 메우고 Phase 3 를 "대(비-임베딩 신설)" 로 확정하는 것은
> 추측 점프다. 지난 7 cycle 의 원칙(처방 규모는 측정이 정한다) 위반.
> Phase 2.5 자동 판정식이 bge-m3 질적 차이를 흡수하도록 설계된 탓 —
> 이 측정이 그 설계 결함을 보정한다.

---

## 1. 측정 과제

### 작업 M1. remedy 스크립트 모델 파라미터화 (최소 패치)

**파일**: `scripts/conflict_recall_remedy_eval.py`

**현재**: `build()` 가 `EmbeddingBridge()` 무인자 호출 (e5-small 고정).

**패치 (이것만, 로직 일체 불변)**: Phase 2.5 의 keyspace 패치와
동일 방식. `build()` 가 model_name 받아 `EmbeddingBridge(model_name=
model_name)` 호출하도록, `main()`/`evaluate()` 진입부에
`sys.argv[1]` → model 주입.

```python
import sys
def build(model_name=None):
    enc = EmbeddingBridge() if model_name is None \
          else EmbeddingBridge(model_name=model_name)
    ...  # 이하 기존 그대로
# evaluate() 안의 build() 호출부에 model 전달
# __main__ 에서: model = sys.argv[1] if len(sys.argv)>1 else None
```

> **절대 불변**: `MARGIN = 0.05`, `calibrate_dist_cut` 의 `95` percentile,
> `score()` 판정식 `h <= 1 and t >= 2`, anchor/probe (fp_eval 에서
> import). 모델 인자 주입 외 **단 한 줄도 바꾸지 말 것.** 재현성 =
> 측정 5/측정 2 와 같은 자(尺) 유지가 이 측정의 전제.

### 작업 M2. 2 모델 대조 실행

| # | 모델 | 목적 |
|---|---|---|
| 1 | (무인자) `intfloat/multilingual-e5-small` | 측정 2 재현 확인 (전부 FAIL 나와야 정상) |
| 2 | `BAAI/bge-m3` | **핵심**: 약한 신호 + 처방 = 분리되는가 |

각각 독립 실행. raw → `docs/03-analysis/conflict_recall_remedy_raw.
{e5-small,bge-m3}.json`.

### 작업 M3. 판정

스크립트 기존 판정식 그대로 (변경 금지):
`PASS = HARD_NEG FP <= 1/6 AND TRUE_POS TP >= 2/2`

**원인·규모 판정 규칙** (e5-small 재현 FAIL 전제하에 bge-m3 결과로):

| bge-m3 처방 결과 | 해석 | Phase 3 규모 |
|---|---|---|
| P1·P2·P3 중 **하나라도 PASS** | 약한 신호를 처방으로 증폭 가능 | **중**: 임베딩 모델 bge-m3 교체 + 해당 처방 레이어 적용. 새 서브시스템 불필요 |
| 전부 FAIL, 단 **bge-m3 의 P1/P2 가 e5-small 대비 HARD_NEG FP 유의 감소** | 신호 있으나 현 처방으로 부족 | **중-대 경계**: 처방 파라미터 탐색(MARGIN/percentile 스윕) 1회 추가 측정 후 재판정 |
| 전부 FAIL, e5-small 과 차이 없음 | bge-m3 양수 마진이 노이즈였음 | **대**: Phase 2.5 결론 확정. 비-임베딩 키(LLM 라벨 추출) 신설 |

> 세 번째 칸이 나와야만 "대" 가 *측정으로* 확정된다. Phase 2.5 는
> 이 측정 없이 "대" 라 했으므로, 이 측정이 그 결론의 진위를 가린다.

---

## 2. 절대 금지 (Phase 1·2·2.5 §금지 계승)

1. ❌ remedy 스크립트의 MARGIN/percentile/판정식/데이터셋 수정 —
   모델 인자 주입 외 일체. 재현성 파괴 = 측정 무효.
2. ❌ 이 결과로 곧장 recall_conflict / EmbeddingBridge 기본모델
   코드 수정 — **측정 지시서다.** 처방은 Phase 3 별도 지시서.
3. ❌ Phase 3 "중/대" 를 측정 전에 단정 — 판정식 출력만 따른다.
4. ❌ xfail 2건(Phase 1·2) GREEN 전환.
5. ❌ LLM 리뷰로 측정 대체. 오직 수치.
6. ❌ bge-m3 가 "부분 신호" 였다는 Phase 2.5 정성 관찰을 근거로
   PASS 를 미리 가정 — 그것을 *검정* 하는 게 이 측정의 목적.

---

## 3. 완료 보고 형식 (표만, 산문 금지)

```
## Phase 2.6 측정 결과

| 모델 | 처방 | HARD_NEG FP | TRUE_POS TP | 판정 |
|------|------|-------------|-------------|------|
| e5-small | Baseline | /6 | /2 | (재현확인) |
| e5-small | P1 / P2 / P3 | ... | ... | FAIL 예상 |
| bge-m3   | Baseline | /6 | /2 | ? |
| bge-m3   | P1 상대마진 | /6 | /2 | ? |
| bge-m3   | P2 분포컷 | /6 | /2 | ? |
| bge-m3   | P3 결합 | /6 | /2 | ? |

e5-small 재현: 측정 2 와 일치 / 불일치(→사유)
bge-m3 최선 처방: (P? / 없음)
원인·규모 판정: 중 / 중-대경계 / 대  (§1.M3 표 근거 명시)
스크립트 패치: (라인 — sys.argv 만)
raw JSON 커밋: (2 경로)
금지 §2 위반: 없음 (필수)
```

`CLAUDE.md` 1줄 append:
`Phase 2.6: bge-m3+처방 = {PASS처방/전부FAIL}. Phase 3 규모 {중/중-대/대} 확정.`

---

## 4. 한 줄 요약 (가장 먼저 읽을 것)

> **코드 안 고친다.** remedy 스크립트에 모델 인자만 추가해 bge-m3
> 로 P1/P2/P3 처방을 돌린다. bge-m3 의 약한 양수 신호(+0.0168)가
> 처방으로 증폭돼 하나라도 PASS 면 Phase 3 = 중(모델 교체+처방).
> 전부 FAIL 이고 e5 와 차이 없으면 Phase 3 = 대(비-임베딩 키).
> Phase 2.5 가 측정 없이 "대" 라 한 그 빈칸을 이 측정이 메운다.
> 측정만 하고 멈춘다.
