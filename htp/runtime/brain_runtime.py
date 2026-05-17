"""
Brain Runtime  —  PFCRuntime + BrainRuntime
=============================================

[Phase 2] 기본 구조:
  - EMA + Cosine Similarity → execute/inhibit
  - 단방향: Thalamus → PFC

[Phase 3] 추가:
  - Working Memory Attention (Scaled Dot-Product)
  - Long-term Goal Alignment (Jaccard-like)
  - Top-down Feedback Loop (PFC → Thalamus 역방향)
  - 결합 점수: 0.6 × cos_sim + 0.4 × goal_score

수학:
  WM Attention:  weights = softmax(Q·K^T / √d),  ctx = Σ w_i·V_i
  refined_vec  = 0.6·current + 0.4·context
  goal_score   = |action_tags ∩ goal_tags| / |goal_tags|
  combined     = 0.6·cos_sim + 0.4·goal_score

생물학:
  - Global Workspace Theory (Baars 1988)
  - Working memory 7±2 (Miller 1956)
  - Top-down attention (Desimone & Duncan 1995)
  - Biased Competition Model
"""

from __future__ import annotations

import re
from collections import deque
from pathlib      import Path
from typing       import Optional, Any

import torch
import torch.nn.functional as F

from .htp_runtime    import HTPRuntime, HTPConfig
from .region_runtime import RegionRuntime
from ..thalamus.region_signal import ThalamusOutput, Action
from ..thalamus.top_down      import TopDownSignal, TopDownBias
from ..memory.memory_system   import MemorySystem


# ══════════════════════════════════════════════════════════
# PFCRuntime
# ══════════════════════════════════════════════════════════

class PFCRuntime(HTPRuntime):
    """
    전전두엽 런타임.

    Phase 3 결정 파이프라인:
      state_vec → WM Attention → refined_vec
      → Cosine Sim (EMA 대비)
      → Goal Alignment (tag 교집합)
      → combined score
      → EXECUTE / INHIBIT + TopDownSignal 생성
    """

    def __init__(self, config: Optional[HTPConfig] = None):
        super().__init__(config)
        self.working_memory       : deque[ThalamusOutput] = deque(maxlen=7)
        self.long_term_goals      : list[str]             = []
        self.inhibition_threshold : float                 = 0.4
        self._ema_vec             : Optional[torch.Tensor]= None
        self._ema_alpha           : float                 = 0.7
        self._wm_lambda           : float                 = 0.4   # WM context 혼합 비율
        self._goal_alpha          : float                 = 0.4   # goal score 가중치
        self._td_computer         : TopDownBias           = TopDownBias()

    # ── 결정 ──────────────────────────────────────────

    def decide(self,
               thal_out: ThalamusOutput,
               regions:  dict = None) -> tuple[Action, TopDownSignal]:
        """
        ThalamusOutput → (Action, TopDownSignal).

        반환값이 tuple로 변경됨 (Phase 3).
        TopDownSignal은 BrainRuntime이 다음 Thalamus.step()에 전달.
        """
        v = thal_out.state_vec

        # 1. Working Memory Attention
        refined_v = self._wm_attention(v)

        # 2. EMA 업데이트
        self._update_ema(refined_v)

        # 3. WM 저장
        self.working_memory.append(thal_out)

        # 4. 결합 정렬 점수
        cos_score  = self._cosine_alignment(refined_v)
        goal_score = self._goal_alignment(thal_out, regions or {})
        score      = (1 - self._goal_alpha) * cos_score + self._goal_alpha * goal_score

        # 5. 결정
        if score >= self.inhibition_threshold:
            action = Action(
                type   = "execute",
                winner = thal_out.winner,
                reason = f"score={score:.3f} (cos={cos_score:.3f} goal={goal_score:.3f})",
            )
        else:
            action = Action(
                type     = "inhibit",
                winner   = thal_out.winner,
                reason   = (f"score={score:.3f} < {self.inhibition_threshold} "
                            f"(cos={cos_score:.3f} goal={goal_score:.3f})"),
                redirect = self._find_redirect(thal_out.winner),
            )

        # 6. TopDownSignal 생성 (다음 스텝 Thalamus에 전달)
        td_signal = self._td_computer.compute(
            goals    = self.long_term_goals,
            regions  = regions or {},
            step     = len(self.working_memory),
            strength = min(score, 1.0),  # 정렬도가 높을수록 강한 top-down
        )

        return action, td_signal

    # ── Working Memory Attention ───────────────────────

    def _wm_attention(self, query: torch.Tensor) -> torch.Tensor:
        """
        Scaled Dot-Product Attention over Working Memory.

        Q = current state_vec
        K = V = past state_vecs in working memory
        weights = softmax(Q·K^T / √d)
        context = Σ w_i·V_i
        output  = (1-λ)·current + λ·context
        """
        if len(self.working_memory) < 2:
            return query

        keys = torch.stack([
            t.state_vec for t in self.working_memory
            if t.state_vec.shape == query.shape
        ])
        if keys.shape[0] == 0:
            return query

        d_k     = float(query.shape[0]) ** 0.5
        scores  = torch.mv(keys, query) / d_k       # [n]
        weights = torch.softmax(scores, dim=0)       # [n]
        context = (weights.unsqueeze(1) * keys).sum(dim=0)  # [d]

        return (1 - self._wm_lambda) * query + self._wm_lambda * context

    # ── Goal Alignment ─────────────────────────────────

    def _goal_alignment(self,
                        thal_out: ThalamusOutput,
                        regions:  dict) -> float:
        """
        Long-term goal tags vs winner Region 발화 태그 교집합.
        score = |goal ∩ action| / |goal|   (Jaccard-like)
        """
        if not self.long_term_goals:
            return 1.0

        goal_tags = set(
            w.lower()
            for g in self.long_term_goals
            for w in g.replace("_", " ").split()
        )

        winner_region = regions.get(thal_out.winner)
        action_tags: set[str] = set()
        if winner_region:
            for n in getattr(winner_region, "_nodes", []):
                action_tags |= getattr(n.fn, "_htp_tags", set())

        return len(goal_tags & action_tags) / max(len(goal_tags), 1)

    # ── Cosine Similarity ──────────────────────────────

    def _cosine_alignment(self, v: torch.Tensor) -> float:
        """현재 vec와 EMA 기준벡터의 Cosine Similarity → [0, 1]."""
        if self._ema_vec is None:
            return 1.0
        v_n = v.norm().item()
        m_n = self._ema_vec.norm().item()
        if v_n < 1e-8 or m_n < 1e-8:
            return 1.0
        cos = F.cosine_similarity(
            v.unsqueeze(0), self._ema_vec.unsqueeze(0)
        ).item()
        return (cos + 1.0) / 2.0

    # ── 내부 ──────────────────────────────────────────

    def _update_ema(self, v: torch.Tensor):
        if self._ema_vec is None or self._ema_vec.shape != v.shape:
            self._ema_vec = v.clone()
        else:
            self._ema_vec = self._ema_alpha * v + (1 - self._ema_alpha) * self._ema_vec

    def _find_redirect(self, current: str) -> str:
        seen: dict[str, int] = {}
        for t in reversed(list(self.working_memory)):
            seen[t.winner] = seen.get(t.winner, 0) + 1
        others = [(w, c) for w, c in seen.items() if w != current]
        return others[0][0] if others else ""


# ══════════════════════════════════════════════════════════
# BrainRuntime
# ══════════════════════════════════════════════════════════

class BrainRuntime:
    """
    최상위 오케스트레이터.

    Phase 3 실행 루프:
      1. 모든 Region 활성화
      2. Thalamus(top_down=prev_td) → ThalamusOutput
      3. PFC.decide() → (Action, new_td)
      4. 억제 피드백 → 패자 Region.apply_suppression()
      5. (선택) Cortico-cortical 약한 신호 전달
      6. new_td를 다음 스텝에 보존

    사용법:
      brain = BrainRuntime()
      brain.add_region("language", lang_region)
      brain.pfc.long_term_goals = ["success", "cache"]
      action = brain.run(data)
    """

    def __init__(self,
                 pfc_config: Optional[HTPConfig] = None,
                 memory_dir: str | Path = ".htp",
                 enable_memory: bool = True,
                 coherence: "object | None" = None):
        """
        coherence: CoherenceStrategy 인스턴스 (None=비활성, 기본).
          비활성 시 기존 동작 그대로 (회귀 보호 — Plan FR-14).
          활성 시 Region 응답 수집 → bind() → conflict 를 swr_priority 증폭.
        """
        self.regions  : dict[str, RegionRuntime] = {}
        self.thalamus                            = None
        self.pfc      : PFCRuntime               = PFCRuntime(pfc_config)
        self._step    : int                      = 0
        self._last_td : Optional[TopDownSignal]  = None
        self._cc                                 = None     # CorticalConnections (선택)
        # [Stage 5-C3] Memory System 연동
        self.memory: Optional[MemorySystem] = (
            MemorySystem(memory_dir=memory_dir) if enable_memory else None
        )
        self._last_state_vec: Optional[torch.Tensor] = None

        # [sub-3 M4] CoherenceGate DI — None 기본 (회귀 보호)
        self.coherence = coherence
        self._last_bound_response = None   # 외부 inspection 용

    def add_region(self, name: str, region: RegionRuntime):
        """Region 추가. Thalamus는 다음 run()에서 재생성."""
        self.regions[name] = region
        self.thalamus      = None

    def enable_cortical_connections(self):
        """Cortico-cortical 연결 활성화. 반환된 객체에 add_connection() 호출."""
        from .cortical_connections import CorticalConnections
        self._cc = CorticalConnections(self.regions)
        return self._cc

    def _ensure_thalamus(self):
        if self.thalamus is None:
            from ..thalamus.thalamus import Thalamus
            self.thalamus = Thalamus(self.regions)

    def run(self, data: Any) -> Action:
        """
        Brain 1 스텝 실행.
        top-down feedback loop + Memory System (Stage 5-C3) 포함.
        """
        self._ensure_thalamus()
        self._step += 1

        # [C3-①] recall: 이전 state_vec 기반 기억 조회
        mem_ctx = None
        if (self.memory is not None
                and self._step > 1
                and self._last_state_vec is not None):
            mem_ctx = self.memory.recall(self._last_state_vec)
            # [C3-②] CA1 추천을 top-down hint 로 주입
            if mem_ctx.recommendation and not mem_ctx.is_novel:
                self._last_td = self._inject_memory_hint(
                    self._last_td, mem_ctx.recommendation, mem_ctx.confidence,
                )

        # 1. 모든 Region 활성화
        for region in self.regions.values():
            if region._nodes:
                try:
                    region.run(data)
                except Exception as e:
                    print(f"  [warn] Region({region.region_name}): {e}")

        # 2. Thalamus (이전 스텝 top-down + memory hint 반영)
        thal_out = self.thalamus.step(data, top_down=self._last_td)

        # [sub-3 M4] CoherenceGate hook (옵션, additive)
        #   - coherence=None 시 기존 동작 (회귀 보호)
        #   - 활성 시 Region 응답 수집 → bind() → _last_bound_response 보존
        #     + conflict_magnitude 를 memory.save() 에 전달 (Plan FR-15)
        self._last_bound_response = None
        if self.coherence is not None:
            bound = self._bind_region_responses()
            if bound is not None:
                self._last_bound_response = bound

        # 3. PFC 결정 + 새 TopDownSignal 생성
        action, td_signal = self.pfc.decide(thal_out, regions=self.regions)

        # 4. top-down 신호 보존
        self._last_td = td_signal

        # [C3-③] state_vec 보존 — 다음 스텝 recall 용
        self._last_state_vec = thal_out.state_vec.detach().clone()

        # [C3-④] 에피소드 저장 (sub-3 M4: conflict_magnitude 전달)
        if self.memory is not None:
            score = self._extract_score(action)
            conflict_magnitude = (
                self._last_bound_response.conflict
                if self._last_bound_response is not None else 0.0
            )
            self.memory.save(
                state_vec          = thal_out.state_vec,
                step               = self._step,
                winner             = action.winner,
                action_type        = action.type,
                score              = score,
                context            = str(data)[:50],
                conflict_magnitude = conflict_magnitude,
            )

        # [C3-⑤] CUSUM overload → consolidation 트리거 (수면 메커니즘)
        if self.memory is not None:
            for name, region in self.regions.items():
                if region._cusum_S > region._cusum_h:
                    self.memory.on_overload(name)
                    region._cusum_S = 0.0       # 수면 후 초기화

        # 5. 억제 피드백 → 패자 Region
        for rid, strength in thal_out.suppressed.items():
            if rid in self.regions and strength > 0:
                self.regions[rid].apply_suppression(strength)

        # 6. Cortico-cortical (활성화된 경우)
        if self._cc is not None:
            self._cc.apply(thal_out)

        # 7. winner Region 결과를 action.result 에 채움
        winner_region = self.regions.get(action.winner)
        if winner_region is not None:
            last = getattr(winner_region, "_last_result", None)
            if last is not None:
                action.result = last.outputs.get(action.winner)

        return action

    # ── CoherenceGate 헬퍼 (sub-3 M4) ─────────────

    def _bind_region_responses(self):
        """모든 Region 의 RegionSignal 을 RegionResponse 로 변환 후 bind.

        차원 통일은 numpy padding/truncate 로 처리. 안전 fallback:
          - Region 수 < 2 → None 반환 (binding 의미 없음)
          - 차원 0 또는 NaN → 해당 Region skip
        """
        if self.coherence is None or len(self.regions) < 2:
            return None
        import numpy as _np
        from ..thalamus.types import RegionResponse

        responses = []
        target_dim = None
        for name, region in self.regions.items():
            try:
                signal = region.collect_signal()
            except Exception:
                continue
            vec = signal.output_vec.detach().cpu().numpy().astype("float64")
            if vec.size == 0 or not _np.isfinite(vec).all():
                continue
            if target_dim is None:
                target_dim = vec.shape[0]
            # 차원 통일 (padding 또는 truncate)
            if vec.shape[0] < target_dim:
                vec = _np.pad(vec, (0, target_dim - vec.shape[0]))
            elif vec.shape[0] > target_dim:
                vec = vec[:target_dim]
            responses.append(RegionResponse(
                region_id  = signal.region_id,
                output_vec = vec,
                precision  = float(signal.precision),
            ))

        if len(responses) < 2:
            return None
        return self.coherence.bind(responses)

    # ── Memory 헬퍼 (Stage 5-C3) ───────────────────

    def _inject_memory_hint(self,
                            td: Optional[TopDownSignal],
                            recommended: str,
                            confidence: float) -> TopDownSignal:
        """CA1 추천 → 기존 td 의 biases[recommended] 를 boost."""
        bias_strength = confidence * 0.5
        if td is None:
            return TopDownSignal(
                biases   = {recommended: bias_strength},
                strength = 0.3,
                step     = self._step,
            )
        td.biases[recommended] = max(
            td.biases.get(recommended, 0.0), bias_strength,
        )
        return td

    @staticmethod
    def _extract_score(action: Action) -> float:
        """action.reason 에서 'score=0.xxx' 파싱."""
        m = re.search(r"score=([-\d.]+)", action.reason or "")
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.5

    def feedback(self, outcome: str, episode_id: Optional[str] = None):
        """행동 결과를 마지막(또는 지정) 에피소드에 기록."""
        if self.memory is not None:
            self.memory.feedback(outcome, episode_id)

    def status(self):
        SEP = "=" * 62
        print(f"\n{SEP}")
        print(f"  BrainRuntime  step={self._step}  regions={len(self.regions)}")
        print(SEP)

        for name, r in self.regions.items():
            try:
                sig = r.collect_signal()
                print(
                    f"  [{name:<12}]  "
                    f"hub={sig.hub_strength:.4f}  "
                    f"fire={sig.fire_rate:.3f}  "
                    f"cusum={r._cusum_S:.2f}  "
                    f"overload={'YES' if sig.overload else 'no '}"
                )
            except Exception as e:
                print(f"  [{name:<12}]  (not built: {e})")

        if self.thalamus:
            print(self.thalamus.status())
            print(self.thalamus.core.report())

        td = self._last_td
        if td and td.biases:
            print(f"\n  [ Top-Down Signal (step={td.step}) ]")
            for rid, b in sorted(td.biases.items()):
                print(f"  {rid:<14}  bias={b:.3f}")

        print(
            f"\n  PFC  wm={len(self.pfc.working_memory)}/7  "
            f"goals={self.pfc.long_term_goals}  "
            f"ema={'ready' if self.pfc._ema_vec is not None else 'init'}"
        )
        print(SEP)
