---
template: design
feature: htp-knowledge-cli-polish
date: 2026-05-17
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
selected_option: B — Clean Modular
selected_id_strategy: UUID 전면 도입
---

# htp-knowledge-cli-polish Design — L2 sidequest

> **Summary**: Stage 0.5 CLI 를 L1 → L2 (매일 쓰는 prototype) 로. **Option B (Clean Modular)** 채택 — `htp/knowledge/cli/` 패키지 + `filters.py` + `exporters.py` 분리. **UUID 전면 도입** — `KnowledgeEntry.id` 필드 신설, 기존 7 entry 는 migration 명령 옵션 제공. Tombstone 패턴으로 delete/edit, JSONL append-only 무손상.
>
> **Selected Architecture**: Option B — Clean Modular
> **Selected ID Strategy**: UUID 전면 도입 (sub-decision #3)
> **Predecessor**: sub-3 (commit `190fc54` — baseline 148)
> **Test Target**: 148 → **155-157** (+7-9)

---

## Context Anchor (Plan 에서 전파)

| Key | Value |
|-----|-------|
| **WHY** | Stage 0.5 가 매일 쓰는 도구가 되어야 v4 Rev 1.3 원칙 완성. sub-3 직후 1-2일 sidequest 의 적정 시점. |
| **WHO** | HTP 개발자 본인. 기존 호출자 (testing/programmatic) 무영향 — UUID 필드는 default_factory 로 자동 부여. |
| **RISK** | (R1) JSONL append-only 깨짐 / (R2) batch encoder.fit() 재호출 / (R3) tombstone 누적 / (R6) L3 사용자 기대 미충족. |
| **SUCCESS** | 회귀 148 유지 + 신규 7-9 tests. S1-S5 모두 Pass. JSONL round-trip 0 손상. |
| **SCOPE** | F1-F5 (batch/stdin/filter/edit/export) 만. L3 (한국어 매칭) + L4 (LLM/Obsidian 통합) OUT. |

---

## 1. Overview

### 1.1 Selected Architecture: Option B — Clean Modular

```
htp/knowledge/
├── __init__.py                  수정 — 공개 export 확장 (filters, exporters)
├── __main__.py                  수정 — `from .cli import main; main()` 로 간소화
├── encoder.py                   무변경
├── loop.py                      수정 — batch ingest / edit / delete 메서드 추가
├── persistence.py               수정 — UUID + tombstone 지원
├── types.py                     [신규] KnowledgeEntry + Tombstone dataclass 분리
├── filters.py                   [신규] source/time/tag 필터 헬퍼
├── exporters.py                 [신규] markdown/json/obsidian 출력
├── migrate.py                   [신규] 기존 jsonl UUID 부여 migration
└── cli/                         [신규 패키지]
    ├── __init__.py              main() dispatch
    ├── ingest.py                ingest (단일 / batch / stdin)
    ├── query.py                 query + filter
    ├── discover.py              discover + filter
    ├── list_cmd.py              list + delete + edit + tag
    └── export.py                export (markdown/json/obsidian)

tests/knowledge/
├── test_loop.py                 수정 — UUID + edit/delete 테스트 추가
├── test_filters.py              [신규]
├── test_exporters.py            [신규]
└── test_cli_dispatch.py         [신규] argparse 통합 smoke
```

### 1.2 Design Goals

| ID | Goal | 측정 방법 |
|----|------|---------|
| G1 | 회귀 148 + 신규 7-9 = **155-157/all** PASS | `pytest -q` |
| G2 | JSONL append-only round-trip 무손상 (delete 후 load_all 미반환) | `test_delete_tombstone_round_trip` |
| G3 | UUID 전면 적용 — 모든 신규 entry id 가 UUID4 | `test_entry_uuid_default` |
| G4 | Batch ingest 시 encoder.fit() 1회만 (옵션 A-2 영속화) | `test_batch_single_fit` |
| G5 | `tests/실사용 테스트.md` S1-S5 Pass | 수동 검증 + 결과 기록 |
| G6 | DAG — `htp/knowledge/cli/* → htp.runtime/thalamus/memory` 미참조 | `test_no_circular_deps.py` parametrize 확장 |

### 1.3 Design Principles

1. **단일 책임** — cli/* 는 argparse + dispatch 만. 로직은 loop/filters/exporters/persistence 위임
2. **회귀 보호** — 기존 KnowledgeEntry/KnowledgeStore 시그니처 유지 (UUID 는 default_factory, 옵셔널)
3. **DAG 단방향** — sub-1 의 knowledge/ DAG 규칙 그대로 유지 (cli/ 도 동일 규칙)
4. **Append-only 보존** — delete/edit 는 마커 line append. 기존 line 절대 수정 안 함

---

## 2. Architecture Detail

### 2.1 KnowledgeEntry 확장 (`htp/knowledge/types.py` 신규)

```python
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class KnowledgeEntry:
    """기존 loop.py 의 dataclass 를 types.py 로 이동 + UUID 추가."""
    text: str
    vec: np.ndarray
    source: str
    timestamp: str
    neighbors: list = field(default_factory=list)
    conflict_count: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))   # 신규
    tags: list[str] = field(default_factory=list)                # 신규


@dataclass
class Tombstone:
    """삭제/수정 마커 — JSONL 에 별도 라인으로 append.

    load_all 시 ref_id 매칭 entry 를 제외 (delete) 또는 대체 (edit).
    """
    kind: str          # "delete" | "edit"
    ref_id: str        # 타깃 entry UUID
    timestamp: str
    replacement_id: str | None = None    # edit 시 새 entry 의 UUID
```

### 2.2 KnowledgeStore 확장 (`persistence.py` 수정)

```python
class KnowledgeStore:
    def append(self, entry: KnowledgeEntry): ...
    def append_tombstone(self, ts: Tombstone): ...    # 신규

    def load_all(self) -> list[KnowledgeEntry]:
        """tombstone 적용된 entry list 반환.

        알고리즘:
          1. 모든 라인 순회 — entry / tombstone 구분
          2. tombstone 적용:
             - delete: ref_id 매칭 entry 제외
             - edit: ref_id 매칭 entry 를 replacement_id 의 entry 로 대체
                     (replacement 가 같은 jsonl 안에 있어야 함)
          3. UUID 없는 legacy entry 는 자동 부여 (in-memory 만)
        """
        ...
```

### 2.3 KnowledgeLoop 확장 (`loop.py` 수정)

```python
class KnowledgeLoop:
    # 기존 ingest(text, source) 그대로 유지 (회귀 보호)

    def ingest_batch(self, texts: list[str], source: str = ""
                    ) -> list[IngestResult]:
        """N 개 텍스트 일괄 ingest — encoder.fit() 1회만."""
        results = []
        for text in texts:
            try:
                results.append(self.ingest(text, source=source))
            except Exception as e:
                # skip-and-continue 정책 (sub-decision #5)
                results.append(IngestError(text=text, error=str(e)))
        return results

    def delete(self, entry_id: str) -> bool:
        """UUID 매칭 entry 삭제 (tombstone)."""
        # _cache 에서 제거 + Tombstone append
        ...

    def edit(self, entry_id: str, new_text: str) -> KnowledgeEntry:
        """edit = delete tombstone + new entry append (id 재부여)."""
        ...

    def add_tags(self, entry_id: str, tags: list[str]) -> KnowledgeEntry:
        """entry 의 tags 필드 확장.

        구현: edit 와 유사하지만 text/vec 유지, tags 만 업데이트.
        """
        ...
```

### 2.4 Filters (`filters.py` 신규)

```python
def filter_entries(
    entries: list[KnowledgeEntry],
    source:  str | None = None,
    since:   str | None = None,        # "Nd" or ISO date
    tag:     str | None = None,
) -> list[KnowledgeEntry]:
    """source/since/tag 필터 적용. None 인자는 무시."""
    out = entries
    if source is not None:
        out = [e for e in out if e.source == source]
    if since is not None:
        cutoff = _parse_since(since)   # datetime + 정규식 (sub-decision #4)
        out = [e for e in out if _parse_ts(e.timestamp) >= cutoff]
    if tag is not None:
        out = [e for e in out if tag in e.tags]
    return out


def _parse_since(spec: str) -> datetime:
    """30d / 2026-04 / 2026-04-15 모두 지원."""
    import re
    if m := re.match(r"^(\d+)d$", spec):
        return datetime.now(timezone.utc) - timedelta(days=int(m[1]))
    # ISO 날짜
    try:
        return datetime.fromisoformat(spec).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"invalid --since: {spec}")
```

### 2.5 Exporters (`exporters.py` 신규)

```python
def export_markdown(entries: list[KnowledgeEntry],
                    group_by: str = "source") -> str:
    """source 별 섹션 + timestamp 정렬."""
    ...

def export_json(entries: list[KnowledgeEntry]) -> str:
    """원본 vec 포함 JSON array."""
    ...

def export_obsidian(entries: list[KnowledgeEntry], dir_path: Path) -> int:
    """파일 단위 split + YAML frontmatter.

    파일명: {timestamp}-{source}-{id_short}.md
    frontmatter: source / tags / created / id
    """
    ...
```

### 2.6 CLI 패키지 (`htp/knowledge/cli/`)

```python
# cli/__init__.py
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _dispatch(args)

def _build_parser() -> argparse.ArgumentParser:
    # 기존 ingest/query/discover + 신규 list/delete/edit/tag/export
    ...

def _dispatch(args) -> int:
    from . import ingest, query, discover, list_cmd, export
    return {
        "ingest":   ingest.run,
        "query":    query.run,
        "discover": discover.run,
        "list":     list_cmd.list_run,
        "delete":   list_cmd.delete_run,
        "edit":     list_cmd.edit_run,
        "tag":      list_cmd.tag_run,
        "export":   export.run,
    }[args.cmd](args)
```

### 2.7 Migration (`migrate.py` 신규)

```python
def migrate_add_uuid(jsonl_path: Path) -> int:
    """기존 jsonl 의 entry 에 UUID 영구 부여.

    절차:
      1. 백업: jsonl_path → jsonl_path.pre-uuid.bak
      2. load_all → UUID 자동 부여 (in-memory)
      3. 새 jsonl 작성 (UUID 포함)
    반환: 마이그레이션된 entry 수
    """
    ...
```

CLI: `python -m htp.knowledge migrate --add-uuid`

---

## 3. DAG 의존 방향

```
htp/knowledge/cli/*  ──→  htp/knowledge/loop.py + filters.py + exporters.py
                      ──→  argparse (stdlib)

htp/knowledge/exporters.py  ──→  htp/knowledge/types.py
htp/knowledge/filters.py    ──→  htp/knowledge/types.py
htp/knowledge/types.py      ──→  numpy / dataclasses / uuid

금지: htp/knowledge/cli/* → htp.runtime / htp.thalamus / htp.memory
        (sub-1 의 knowledge DAG 규칙 그대로 유지)
```

`test_no_circular_deps.py` 의 `_KNOWLEDGE_DIR.glob("*.py")` 가 자동으로 cli/ 하위 파일도 검사하도록 재귀 옵션 추가.

---

## 4. Session Plan

| Session | Scope | 누적 테스트 | 소요 |
|---------|-------|----------|------|
| **session-1** | types.py 분리 + UUID + persistence tombstone + migration | 148 → 151 | ~3-4h |
| **session-2** | loop.py 확장 (batch/delete/edit/tag) + filters.py | 151 → 154 | ~3-4h |
| **session-3** | cli/ 패키지 + exporters.py + DAG 확장 | 154 → 157 | ~3-4h |

총 ~10-12h.

---

## 5. Test Plan (+7-9 신규)

### 5.1 회귀 보호 (148)

- 기존 ingest/query/discover 호출 시그니처 무변경
- 기존 7 entry 자료 load_all → UUID 자동 부여 (in-memory) + 그 외 동작 동등
- encoder_state.pkl 영속화 무영향

### 5.2 신규 (tests/knowledge/)

| ID | 테스트 | 파일 | Stage |
|----|------|------|------|
| T1 | `test_entry_uuid_default` | test_loop.py | UUID #3 |
| T2 | `test_batch_single_fit` | test_loop.py | F1 / R2 |
| T3 | `test_batch_skip_and_continue` | test_loop.py | F1 sub-decision #5 |
| T4 | `test_delete_tombstone_round_trip` | test_loop.py | F4 / FR-18 G2 |
| T5 | `test_edit_id_unique` | test_loop.py | F4 |
| T6 | `test_filter_source_since_tag` | test_filters.py | F3 |
| T7 | `test_export_markdown_grouped` | test_exporters.py | F5 |
| T8 | `test_export_json_round_trip` | test_exporters.py | F5 |
| T9 | `test_cli_dispatch_smoke` | test_cli_dispatch.py | All |

### 5.3 수동 검증 (`tests/실사용 테스트.md`)

S1-S5 시나리오 모두 실행 + 결과 기록. 실제 7 entry 자료에 30+ 신규 entry 추가하며 검증.

---

## 6. 8 Sub-Decision 최종 결정

| # | 항목 | 결정 | 근거 |
|---|------|------|------|
| 1 | Tombstone format | **별도 marker line** (`Tombstone` dataclass) | append-only 보존 + load_all 단순 |
| 2 | Tags schema | **`list[str]`** | Python/JSON 자연스러움 |
| 3 | Edit id 방식 | **UUID 전면 도입** (사용자 선택) | 모든 entry UUID4 + migration 명령 옵션 |
| 4 | `--since` 파싱 | **stdlib datetime + 정규식** | 의존성 추가 회피 |
| 5 | Batch partial failure | **skip-and-continue** | Plan S1 Pass 기준 |
| 6 | Architecture | **Option B Clean Modular** (사용자 선택) | cli/ 패키지 + filters/exporters 분리 |
| 7 | CLI dispatch 위치 | **`cli/__init__.py`** | argparse + dispatch 만 |
| 8 | Migration | **별도 `migrate.py` + `--add-uuid` 명령** | 선택적 적용 가능, 기본은 in-memory UUID |

---

## 7. Risks + Mitigations

| ID | Risk | Mitigation |
|----|------|----------|
| R1 | JSONL append-only 패턴 깨짐 (tombstone 실수) | T4 `test_delete_tombstone_round_trip` + 기존 7 entry 백업 (`.htp/knowledge_log.pre-cli-polish.jsonl`) |
| R2 | Batch encoder.fit() 재호출 | 옵션 A-2 영속화 `_fitted` 가드 + T2 회귀 보호 |
| R3 | Tombstone 누적 비대화 | sub-cycle 가정 N<1000. `compact` 명령은 후속 |
| R4 | UUID 도입으로 기존 호출자 깨짐 | `id` 는 default_factory 로 자동 부여 — 기존 코드 무영향 |
| R5 | cli/ 패키지 분할로 회귀 | session-3 에서 한꺼번에 cli/ 도입, 각 step 직후 pytest |
| R6 | L3 (한국어 매칭) 미충족 (Plan §6 R6) | 명시적 OUT-OF-SCOPE — Report 에 기록 |

---

## 8. Decision Record

| Decision | Choice | Rationale |
|----------|--------|----------|
| Architecture | **Option B Clean Modular** | 사용자 선택. cli/ 패키지 + filters/exporters 분리. 7-8 파일 추가, 향후 확장 용이 |
| Edit id 방식 | **UUID 전면 도입** | 사용자 선택. backward-compat 일부 손실 vs stable ref. migration 명령 보완 |
| Tombstone format | 별도 marker line | Append-only 강제 + load_all 단순화 |
| Tags schema | `list[str]` | Python/JSON 자연스러움 |
| `--since` 파싱 | stdlib + 정규식 | 의존성 0 |
| Batch failure | skip-and-continue | S1 Pass 기준 |
| Migration | `python -m htp.knowledge migrate --add-uuid` | 옵셔널 — 기존 사용자 강제 변경 X |

---

## 9. Out-of-Scope (sub-cycle)

- 한국어 의미 매칭 (S6, S7) — **sub-5 EmbeddingBridge 자동 해결**
- Obsidian sync (S8) — `htp-knowledge-integration`
- LLM query 통합 (S9) — `htp-knowledge-integration`
- Daily 자동화 (S10) — `htp-knowledge-integration`
- Tombstone compaction (누적 JSONL 정리) — 후속 사이클
- HTP core (thalamus / memory / runtime) 변경

---

## 10. Checkpoint Summary

- **Architecture**: ✅ Option B — Clean Modular
- **ID Strategy**: ✅ UUID 전면 도입
- **Session 분할**: ✅ 3 sessions (types+UUID / loop+filters / cli+exporters)
- **테스트 목표**: 148 → **155-157**
- **다음 액션**: `/pdca do htp-knowledge-cli-polish --scope session-1`
