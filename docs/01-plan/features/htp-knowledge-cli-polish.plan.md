---
template: plan
version: 1.0
feature: htp-knowledge-cli-polish
date: 2026-05-17
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# htp-knowledge-cli-polish Planning Document

> **Summary**: Stage 0.5 Knowledge Loop 의 CLI 사용성을 L1 (가설 검증용, 7-entry 수준) → **L2 (매일 쓰는 prototype)** 으로 끌어올리는 sidequest. 5개 CLI 기능 (batch / stdin / filter / edit / export) 추가. **TF-IDF 한국어 의미 매칭 한계는 본 사이클 OUT-OF-SCOPE** — sub-5 (Stage 6 EmbeddingBridge) 가 자동 해결 (L3).
>
> **Project**: HTP
> **Version**: post-`190fc54` (htp-thalamus-car sub-3 완료, baseline 148)
> **Author**: Mindbuild
> **Date**: 2026-05-17
> **Status**: Draft
> **선행 문서**:
> - `TODO.md` (백로그 항목 정의)
> - `tests/실사용 테스트.md` (S1-S5 시나리오 + Pass 기준)

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | 현재 `python -m htp.knowledge {ingest,query,discover}` 3 subcommand 만으로는 매일 사용 불가능. (1) 매번 1개 텍스트 ingest — 30 파일 처리 시 30번 프로세스 spawn + encoder cache 재로드 비효율. (2) source/time/tag filter 없음 — 누적 시 검색 정확도 저하. (3) 잘못 저장된 entry edit/delete 불가 (append-only). (4) 다른 도구로 export 불가. |
| **Solution** | 5 CLI 기능 추가: **(F1) Batch ingest** (`--file` / `--dir`), **(F2) Stdin pipe** (인자 미존재 시 stdin), **(F3) Filter** (`--source` / `--since` / `--tag`), **(F4) Edit/delete/tag** (tombstone 패턴), **(F5) Export** (markdown / json / obsidian). JSONL append-only 깨지 않고 tombstone 으로 delete 구현. encoder state 1회 fit 보장. |
| **Function/UX Effect** | 매일 사용 가능 — Obsidian Daily Notes 30일치 일괄 ingest / 회의 메모 stdin pipe / 한 달치 markdown export. **L1 → L2 분기점 도달**. 신규 테스트 5-7개. 100+ entry 운용 가능. |
| **Core Value** | "Stage 0.5 가 매일 쓰는 도구가 되어야 v4 Rev 1.3 원칙 ('루프를 먼저 닫는다') 이 완성된다". sub-1 직후 7-entry 실험으로 가설 검증, 이번 사이클로 실사용 prototype 진입. sub-5 EmbeddingBridge 진입 전 사용자 피드백을 누적할 운영 토대. |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | TODO.md 에 명시 — "0.5 가 매일 쓰는 도구 분기점 (L2) 도달은 sub-3 직후 1-2일 sidequest". 현재 시점 도달. |
| **WHO** | 사용자 본인 (HTP 개발자 + Knowledge Loop 일상 사용자). 기존 호출자 (testing / programmatic) 무영향 — 신규 subcommand 만 추가. |
| **RISK** | (1) JSONL append-only 패턴 깨짐 → tombstone 안전 처리 필요. (2) batch 시 encoder.fit() 재호출 가능성 → 옵션 A-2 영속화 활용. (3) export 포맷 다양성 → 핵심 3종 (markdown/json/obsidian) 만. (4) **L3 (한국어 의미 매칭) 은 OUT-OF-SCOPE** — sub-5 자동 해결, 본 사이클 미해결 가능성 명시. |
| **SUCCESS** | (1) 회귀 148/148 유지 + 신규 5-7 tests. (2) `tests/실사용 테스트.md` S1-S5 의 Pass 기준 모두 충족. (3) 7 entry 자료에 30+ 신규 entry 추가 + filter/edit/export 모두 작동. (4) JSONL 손상 0건. |
| **SCOPE** | F1-F5 5 기능만. **F6 (한국어 의미 매칭, S6/S7)** 는 sub-5 OUT. F7+ (LLM 통합, Obsidian sync, daily 자동화 = S8-S10) 는 별도 cycle `htp-knowledge-integration`. |

---

## 1. Overview

### 1.1 Background

- **Stage 0.5 (sub-1) 완료 시 7-entry baseline** — gas 검증용으로 충분, 매일 사용 prototype 으로는 부족
- **분기점 4단계 (TODO.md 명시)**:
  - L1 (현재): 가설 검증용 (sub-1 완료 시점)
  - **L2 (이 사이클)**: 매일 쓰는 prototype
  - L3 (sub-5 자동): 한국어 의미 매칭 신뢰
  - L4 (별도): Obsidian/LLM 통합 완성형

### 1.2 Why this cycle, why now

- sub-3 (CoherenceGate) 완료 → Thalamus routing 진도 충분
- sub-4 (LLMRegion) 진입 전 0.5 의 실사용 피드백 누적 시작 가치 큼
- 1-2일 소요로 sub-4 진입 일정에 영향 미미
- v4 Rev 1.3 원칙 ("매일 쓰는 도구가 되어야 가치") 가 본 사이클로 완성

### 1.3 Out-of-Scope (명시)

- 한국어 의미 매칭 정확도 개선 (S6, S7) — **sub-5 EmbeddingBridge 자동 해결**
- Obsidian 양방향 sync (S8) — `htp-knowledge-integration` cycle
- LLM query 통합 (S9) — `htp-knowledge-integration` cycle
- Daily 자동화 (S10) — `htp-knowledge-integration` cycle
- HTP core (thalamus / memory / runtime) 의 변경 — 본 사이클 IN-SCOPE 아님

---

## 2. Requirements (FR)

| ID | Stage | Requirement | Priority | Status |
|----|------|-------------|---------|------|
| **FR-01** | F1 | `ingest --file <path>` — 단일 파일 ingest, 전체 내용을 1 entry 로 | High | Pending |
| **FR-02** | F1 | `ingest --dir <path> --pattern "*.md"` — 디렉토리 일괄, 파일당 1 entry | High | Pending |
| **FR-03** | F1 | Batch 시 encoder.fit() 1회 보장 (옵션 A-2 영속화 활용) | High | Pending |
| **FR-04** | F1 | 진행률 표시 (`[N/M] file.md ✓`) + 실패 skip + 마지막 에러 요약 | Med | Pending |
| **FR-05** | F2 | `ingest --source X` (text 인자 없음) → stdin 사용 | High | Pending |
| **FR-06** | F2 | 빈 stdin / non-TTY 처리 안전성 (명확한 에러 메시지) | Med | Pending |
| **FR-07** | F3 | `query/discover --source X` — source 매칭만 검색 | High | Pending |
| **FR-08** | F3 | `query/discover --since 30d` 또는 `--since 2026-04` — 시간 필터 | High | Pending |
| **FR-09** | F3 | `query/discover --tag X` — 태그 필터 (FR-12 tag 지원과 함께) | Med | Pending |
| **FR-10** | F4 | `list --source X --limit N` — entry id + summary 표시 | Med | Pending |
| **FR-11** | F4 | `delete --id N` — tombstone 패턴 (JSONL 손상 0건) | High | Pending |
| **FR-12** | F4 | `tag --add "..." --id N` — entry 에 태그 추가 (`tags` 필드 신설) | Med | Pending |
| **FR-13** | F4 | `edit --id N --text "..."` — 본문 수정, timestamp + vec 재계산 (id 유지) | Med | Pending |
| **FR-14** | F5 | `export --format markdown --source X --since Y` — markdown 출력 | High | Pending |
| **FR-15** | F5 | `export --format json` — 원본 vec 포함 JSON | Med | Pending |
| **FR-16** | F5 | `export --format obsidian --dir <path>` — 파일 단위 split + frontmatter | Low | Pending |
| **FR-17** | All | 회귀 148/148 유지 + 신규 5-7 tests | High | Pending |
| **FR-18** | All | JSONL append-only + tombstone 의미 무손상 (round-trip 보호) | High | Pending |

---

## 3. Design Constraints

| Constraint | Detail |
|-----------|--------|
| **Append-only 보존** | 기존 entry 의 vec/text 직접 수정 금지. delete = tombstone marker append. edit = old tombstone + new entry append. |
| **encoder.fit() 1회** | Critical Gap #3 옵션 A-2 영속화 활용. batch 시 `.htp/encoder_state.pkl` 로 fit state 공유. |
| **회귀 보호** | 기존 3 subcommand (ingest/query/discover) 의 시그니처 완전 보존. 신규 인자는 모두 옵셔널. |
| **단일 책임 분리** | CLI dispatch (`__main__.py`) 는 argparse 만, 로직은 `loop.py` / `persistence.py` 에 위임 |
| **테스트 격리** | tempdir 기반 KnowledgeStore 사용 (sub-1 패턴 재사용) |

---

## 4. Success Criteria

### 4.1 정량 (회귀 + 신규 테스트)

| 지표 | Before | After (목표) |
|------|------:|-------:|
| 총 테스트 | 148 | **153-155** (+5-7) |
| 회귀 깨짐 | 0 | **0** |
| JSONL 손상 round-trip 테스트 | 없음 | **+1** (FR-18) |
| Batch ingest 처리량 | 1 file/process | **N files / 1 process** |

### 4.2 정성 (시나리오)

`tests/실사용 테스트.md` 의 S1-S5 모두 Pass 기준 충족:
- **S1** Batch ingest (Obsidian Daily 30일치)
- **S2** Stdin pipe (회의 메모 받아쓰기)
- **S3** Source/time filter (지난 한 달 brain 도메인)
- **S4** Edit/delete/tag (잘못 저장 entry 삭제 + 사후 태그)
- **S5** Export (한 달치 markdown)

---

## 5. Implementation Sketch

### 5.1 File map

```
htp/knowledge/
├── __main__.py         수정 — argparse 확장 (5 신규 subcommand + 옵션)
├── loop.py             수정 — ingest 가 list[str] 받아 batch 처리 + edit/delete
├── persistence.py      수정 — tombstone marker 지원 + load_all 시 tombstone 적용
├── filters.py          신규 — source/time/tag 필터 헬퍼
└── exporters.py        신규 — markdown/json/obsidian 출력 헬퍼

tests/knowledge/
├── test_loop.py        수정 — 신규 batch/edit/delete 테스트 추가
├── test_filters.py     신규 — filter unit
└── test_exporters.py   신규 — export round-trip
```

### 5.2 Session Plan

| Session | Scope | 누적 테스트 | 소요 |
|---------|-------|----------|------|
| **session-1** F1+F2 | Batch ingest + stdin pipe | 148 → 151 | ~3-4h |
| **session-2** F3+F4 | Filter + Edit/delete/tag | 151 → 154 | ~4-5h |
| **session-3** F5 | Export (markdown/json/obsidian) | 154 → 155+ | ~2-3h |

총 ~10-12h (1-2일).

---

## 6. Risks

| ID | Risk | Severity | Mitigation |
|----|------|----------|----------|
| R1 | JSONL append-only 깨짐 (delete/edit 구현 실수) | High | Tombstone 패턴 + FR-18 round-trip 테스트 + 기존 7 entry 백업 (`.htp/knowledge_log.pre-cli-polish.jsonl`) |
| R2 | Batch 시 encoder.fit() 재호출로 임베딩 공간 불일치 | Med | 옵션 A-2 영속화 `_fitted` 가드 그대로 활용. 단 첫 batch 시 fit 코퍼스 크기 trade-off (작은 vocab → 새 어휘 미반영) 는 sub-5 까지 수용 |
| R3 | Tombstone 누적으로 JSONL 비대화 | Low | sub-1 의 sub-5 까지는 N < 1000 가정. compaction (`htp.knowledge compact`) 은 후속 사이클 |
| R4 | Time filter 의 timezone 해석 모호 | Low | UTC 고정 (sub-1 영속화 timestamp 가 ISO+TZ) + `--since 30d` 는 now() 기준 |
| R5 | Obsidian export 의 frontmatter 호환성 | Low | Obsidian 표준 YAML frontmatter (source/tags/created) 만. 고급 plugin 호환성은 OUT |
| R6 | **L3 (한국어 매칭) 사용자 기대 미충족** | Med | **명시적 문서화** — 본 사이클 OUT, sub-5 EmbeddingBridge 가 본질 해결 |

---

## 7. Test Strategy

### 7.1 회귀 보호 (148/148)

- 기존 ingest/query/discover 3 subcommand 시그니처 무변경
- 기존 7 entry 자료 (`.htp/knowledge_log.jsonl`) 로드 round-trip
- encoder_state.pkl 영속화 무영향

### 7.2 신규 테스트 (+5-7)

| ID | 테스트 | Stage |
|----|------|------|
| T1 | `test_batch_ingest_file` — 단일 파일 ingest | F1 |
| T2 | `test_batch_ingest_dir_pattern` — 디렉토리 일괄 + 진행률 | F1 |
| T3 | `test_stdin_ingest_safe` — 빈 stdin 안전 에러 | F2 |
| T4 | `test_filter_source_and_time` — source/since 조합 | F3 |
| T5 | `test_delete_tombstone_round_trip` — delete → load_all 시 미반환 (FR-18) | F4 |
| T6 | `test_edit_id_preserved` — edit 시 id 유지 + vec 재계산 | F4 |
| T7 | `test_export_markdown_grouped` — source 별 섹션 + timestamp 정렬 | F5 |

### 7.3 수동 검증 (`tests/실사용 테스트.md`)

S1-S5 시나리오 Pass 기준을 수동으로 한 번 실행 + 결과 기록 (사용자 실사용 시나리오).

---

## 8. Decisions to Defer (sub-decision 필요)

Design 단계에서 결정할 사항:
1. **Tombstone format** — 별도 marker line vs entry 내 `deleted: true` 필드
2. **Tags 필드 schema** — list[str] vs string CSV
3. **Edit 의 id 유지 방식** — UUID 도입 vs 라인 번호 그대로
4. **Filter 의 `--since` 파싱 라이브러리** — 표준 stdlib datetime vs dateutil (의존성 추가)
5. **Batch 처리 시 partial failure 정책** — fail-fast vs skip-and-continue

---

## 9. Next Steps

1. **Design 진입**: `/pdca design htp-knowledge-cli-polish`
   - 8개 sub-decision 결정 + 3-architecture options 제시
2. **Do**: Session 3분할 (F1+F2 / F3+F4 / F5)
3. **Check + Report**: sub-1/2/3 패턴 재사용
