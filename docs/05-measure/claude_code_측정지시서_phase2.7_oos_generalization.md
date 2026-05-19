# Claude Code 측정 지시서 — Phase 2.7: 분포컷 일반화 검정 (out-of-sample)

> **출처**: 컨테이너 실측 6회 + Phase 2.5(keyspace) + Phase 2.6(bge-m3 처방) 완료.
> **성격**: 측정 지시서. 코드 수정 아님. Phase 3 규모를
>          **중(모델교체+고정컷)** vs **대(동적 재calibrate 메커니즘 신설)** 로
>          *최종* 확정하는 측정. Phase 2.6 의 PASS 가 일반화되는지 검정.
> **원칙**: 측정값이 정한다. Phase 2.6 PASS 의 두 약점을 닫는다.

---

## 0. 왜 이 측정인가 — Phase 2.6 PASS 의 두 약점

Phase 2.6 은 `bge-m3 + P2 분포컷(cos≥0.621)` 으로 PASS
(TP 2/2, HARD_NEG FP 1/6) → "Phase 3 = 중, 서브시스템 불필요" 로 닫음.
**이 PASS 는 두 가지 이유로 Phase 3 확정 근거로 불충분하다.**

### 약점 A — 표본이 너무 작다

- TRUE_POS = **단 2건**. "재현율 100%" 의 분모가 2.
  동전 두 번 던져 앞면 2번과 통계적으로 동급.
- HARD_NEG FP = **1/6, 합격선 ≤1 에 정확히 걸림.**
  샘플 1개만 다른 방향이었으면 FAIL. 경계선 한 칸 통과.
- → 첫 외부 리뷰 §1 의 "헤드라인-표본 비대칭" 이 측정 설계
  안으로 재유입. PASS 가 신호인지 표본운인지 미분리.

### 약점 B — 0.621 컷이 in-sample 과적합

- `calibrate_dist_cut()` 는 그 자리의 NEG 12 probe 의 cos 95th
  percentile 로 0.621 산출. **즉 컷이 시험 답안(그 NEG)을 보고
  역산됨.** 그 NEG 를 1/6 거른 건 당연 — leakage.
- 운영에서는 새 충돌의 cos 를 미리 못 보므로 그렇게 못 정함.
  0.621 = in-sample 최적값, **out-of-sample 일반화 미측정.**
- → §6 Bug 2 의 "threshold 0.6 시연데이터 하드코딩 → 일반입력
  거짓양성 100%" 와 **구조 동일.** 같은 실수 반복 위험.

> Phase 2.6 의 §1.M3 판정식이 표본크기·누수를 구분 못 하도록
> 설계된 게 근본 원인 (Phase 2.5 자동판정식이 bge-m3 질적차이를
> 흡수한 것과 같은 종류의 설계 결함, 2회 반복). 이 측정이 보정.

---

## 1. 측정 과제

### 작업 M1. 데이터셋 확장 — TRUE_POS 표본 + holdout 셋

**파일**: `scripts/conflict_recall_fp_eval.py` 의 `ANCHORS` / `PROBES`

현재: anchor 6, TRUE_POS 2, NEG 12. → 확장:

**M1-a. anchor 6 → 그대로 유지** (재현성). 신규 anchor 추가 금지.

**M1-b. TRUE_POS 2 → 최소 10 으로 확대.**
기존 anchor 6 각각에 대해 "같은 충돌의 다른 표면 표현" probe 를
추가. anchor 당 1~2 개씩, 총 ≥10. 작성 규칙:
- 같은 추상 충돌, **완전히 다른 어휘·도메인 표면**
  (예: anchor0 'Redis LRU eviction' → TP 'OS 페이지 교체 알고리즘
   clock sweep', 'CDN 엣지 캐시 만료 정책')
- 기존 2개 (LRU 축출 / 셀프어텐션) 는 유지하고 추가만.

**M1-c. NEG 도 holdout 분리 가능하게 확대.**
HARD_NEG 6 → 12 로 (anchor 당 2개). 신규 6개는 기존과 같은
"같은 도메인·다른 메커니즘" 규칙.

**M1-d. split 태그 추가.** 각 probe 에 `"calib"` 또는 `"eval"`
태그. **calib = 컷 산출 전용, eval = 평가 전용. 겹치지 않음.**
- calib: NEG 의 절반 (컷 calibrate 입력)
- eval : 나머지 NEG 절반 + **모든 TRUE_POS** (한 번도 컷 산출에
  안 쓰인 데이터로만 성능 측정)

> 데이터 작성은 LLM 생성이지만, **이것은 측정 대상이 아니라
> 측정 도구**다. 작성 후 사람(사용자)이 TP/HARD_NEG 라벨 타당성을
> 검수하는 단계가 Phase 2.7 실행 전 1회 필요 (§3 보고에 명시).

### 작업 M2. out-of-sample 프로토콜로 remedy 재실행

**파일**: `scripts/conflict_recall_remedy_eval.py` — **로직 변경
최소화.** 추가 패치는 split 분리뿐:

```
1. calib 태그 probe 의 NEG cos 분포 → 95th pct 로 컷 산출
   (= 기존 calibrate_dist_cut, 입력만 calib 로 제한)
2. 그 컷을 고정.
3. eval 태그 probe 에만 적용해 TP 재현율 / HARD_NEG FP 측정
   (eval 은 컷 산출에 1도 안 쓰임 = 진짜 out-of-sample)
```

> **불변**: MARGIN=0.05, percentile=95, score 판정식 `h<=1 and
> t>=2` 의 *형태*. 단 분모가 커지므로 합격선은 비율로 재해석
> (§M3). anchor/판정 로직 일체 무변경. split 라우팅만 추가.

모델: `BAAI/bge-m3` 단일 (Phase 2.6 에서 e5 는 신호 없음 확정,
재현 불요). baseline 대조로 e5-small 1회만 병기.

### 작업 M3. 판정 — 비율 기준 + 일반화 격차

분모가 커졌으므로 절대수 아닌 비율로:

| 지표 | 합격 |
|---|---|
| eval TRUE_POS 재현율 | ≥ 80% (≥8/10) |
| eval HARD_NEG 거짓양성률 | ≤ 20% (≤2~3/12) |
| **calib→eval 일반화 격차** | 재현율/FP 가 calib 대비 **악화 ≤ 15%p** |

**규모 판정**:

| eval 결과 | 해석 | Phase 3 |
|---|---|---|
| 재현율 ≥80% AND FP ≤20% AND 격차 ≤15%p | 0.621류 컷이 일반화됨 | **중**: bge-m3 교체 + **고정** 분포컷. 서브시스템 불필요. Phase 2.6 결론 *측정으로* 확정 |
| 재현율/FP 는 통과하나 격차 >15%p | 신호 있으나 컷이 데이터의존 | **중-대**: 분포컷을 운영 중 주기적 재calibrate 하는 경량 메커니즘 필요 (서브시스템은 아님) |
| eval 재현율 <80% 또는 FP >20% | in-sample PASS 는 과적합이었음 | **대**: 고정컷·동적컷 모두 불충분. 비-임베딩 키(LLM 라벨) 신설. Phase 2.6 "중" 반증 |

> 셋째 칸이 나오면 Phase 2.6 의 "서브시스템 불필요" 가 측정으로
> 반증되고 Phase 2.5 의 "대" 가 (다른 경로로) 부활. 첫째 칸이
> 나와야만 "중" 이 *일반화 검정을 통과한* 확정이 된다.

---

## 2. 절대 금지 (Phase 1·2·2.5·2.6 계승)

1. ❌ anchor 6 변경/추가, MARGIN·percentile·score 형태 변경.
   확장은 TRUE_POS/HARD_NEG probe 추가 + split 태그 + split
   라우팅 **뿐.**
2. ❌ calib 와 eval probe 중복 — leakage 재발. 분리 검증 필수.
3. ❌ 결과로 곧장 EmbeddingBridge default 모델 / recall_conflict
   수정 — 측정 지시서다. 처방은 Phase 3.
4. ❌ Phase 3 "중/대" 측정 전 단정. 판정식 출력만.
5. ❌ xfail 2건 GREEN 전환.
6. ❌ 신규 TRUE_POS/HARD_NEG 를 "쉽게" 작성해 PASS 유도 —
   HARD_NEG 는 *같은 도메인 다른 메커니즘* 난이도 유지. 검수 단계
   필수.
7. ❌ LLM 리뷰로 측정 대체.

---

## 3. 완료 보고 형식 (표만)

```
## Phase 2.7 측정 결과

데이터셋: anchor 6 / TRUE_POS {N≥10} / HARD_NEG {N≥12}
split: calib NEG {n} / eval NEG {n} + TRUE_POS 전수
라벨 검수: 사용자 검수 완료 여부 (Y/N — N 이면 측정 보류)

| 모델 | split | TRUE_POS 재현율 | HARD_NEG FP율 | 판정 |
|------|-------|-----------------|---------------|------|
| bge-m3 | calib (참고) | x/x (xx%) | x/x (xx%) | - |
| bge-m3 | **eval** | x/10 (xx%) | x/12 (xx%) | PASS/FAIL |
| e5-small | eval (대조) | ... | ... | FAIL 예상 |

산출 컷(calib 기준): cos≥0.xxx
calib→eval 일반화 격차: 재현율 -x%p / FP +x%p
규모 판정: 중 / 중-대 / 대  (§1.M3 근거 명시)
스크립트 패치: fp_eval(데이터+split) / remedy_eval(split 라우팅) 라인
raw JSON: (경로)
금지 §2 위반: 없음 (필수)
```

`CLAUDE.md` 1줄:
`Phase 2.7: bge-m3 분포컷 out-of-sample = {일반화/과적합}. Phase 3 규모 {중/중-대/대} 최종확정.`

---

## 4. 한 줄 요약 (가장 먼저 읽을 것)

> **코드 안 고친다.** TRUE_POS 를 2→≥10 으로 늘리고, NEG 를
> calib/eval 로 분리해, 컷은 calib 으로만 산출, 성능은 eval 로만
> 측정한다(out-of-sample). eval 재현율 ≥80% + FP ≤20% + 일반화
> 격차 ≤15%p 면 Phase 3 = 중(고정컷 확정). eval 에서 무너지면
> 0.621 은 과적합이었고 Phase 3 = 대(비-임베딩 키). 신규 데이터는
> 사용자 라벨 검수 후 측정. 측정만 하고 멈춘다.
