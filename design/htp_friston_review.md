# HTP — Karl Friston Free Energy Principle 기반 수정 설계

## 핵심 원리

> "뇌의 모든 것은 하나의 원리로 귀결된다 — 자유 에너지 최소화(Free Energy Minimization)"
> — Karl Friston

```
자유 에너지 (Variational Free Energy):
  F = E_Q[log Q(s) - log P(o, s)]
    = 예측 오차 에너지 + KL 복잡도
    = -log P(o | model) + KL[Q(s) || P(s)]

뇌가 하는 모든 것:
  지각  → F 최소화 (모델 업데이트)
  행동  → F 최소화 (환경 변경)
  주의  → 정밀도(precision) 조절
  수면  → 모델 통합 및 압축
```

---

## 비판 → 수정 매핑

| 비판 | 현재 코드 | FEP 수정 |
|------|------|------|
| 반응 기계 (예측 없음) | Region이 입력만 처리 | PredictiveRegion 추가 |
| 고정 임계값 | `threshold=0.35` 하드코딩 | precision 기반 동적 임계값 |
| 코사인 유사도 결정 | `_cosine_alignment()` | Variational Free Energy |
| 시상 = 단순 게이트 | Sigmoidal Gate | Precision-weighted Gate |
| score 기반 행동 선택 | `score >= threshold` | Expected Free Energy 최소화 |
| 발화 집중도 과부하 | Shannon Entropy CUSUM | 예측 오차 누적 CUSUM |
| 태그 교집합 top-down | Jaccard 유사도 | Softmax 확률 분포 prior |

---

## 즉시 수정 (Phase 2)

---

### 수정 1: RegionSignal에 precision 필드 추가

**현재:**
```python
# region_signal.py
@dataclass
class RegionSignal:
    region_id:    str
    hub_strength: float
    fire_rate:    float
    top_hubs:     list
    overload:     bool
    output_vec:   torch.Tensor
    # precision 없음!
```

**수정:**
```python
@dataclass
class RegionSignal:
    region_id:    str
    hub_strength: float
    fire_rate:    float
    top_hubs:     list
    overload:     bool
    output_vec:   torch.Tensor
    precision:    float = 1.0      # ★ 추가: 예측 오차 신뢰도
    prediction_error: float = 0.0  # ★ 추가: 현재 예측 오차 크기
```

---

### 수정 2: RegionRuntime — precision 동적 계산

**현재:**
```python
# region_runtime.py
# precision 개념 없음
# threshold = cfg.threshold (고정)
```

**수정:**
```python
# region_runtime.py

class RegionRuntime(HTPRuntime):
    def __init__(self, region_name, specialty, config=None, gen_config=None):
        super().__init__(config)
        ...
        # FEP: precision 상태
        self._precision:          float = 1.0   # ★ 추가: 예측 오차 분산의 역수
        self._predicted_vec:      torch.Tensor | None = None  # ★ 예측 벡터
        self._prediction_errors:  list[float] = []            # ★ 오차 이력

    @property
    def dynamic_threshold(self) -> float:
        """
        FEP: 정밀도 기반 동적 발화 임계값
        
        수학:
          threshold_i = base_threshold / precision_i
          
          precision이 높음 (신뢰할 수 있는 입력)
            → threshold 낮아짐 → 더 민감하게 발화
          precision이 낮음 (노이즈가 많은 입력)
            → threshold 높아짐 → 발화 억제
        
        생물학:
          시상-피질 루프에서 정밀도가 높은 감각 채널이
          더 낮은 임계값으로 피질에 투영됨
        """
        return self.cfg.threshold / (self._precision + 1e-8)

    def update_precision(self, prediction_error: torch.Tensor):
        """
        예측 오차로부터 precision 업데이트 (EMA)
        
        수학:
          var_t = var(prediction_error)
          precision_t = 1 / var_t
          precision_ema = 0.9 × precision_prev + 0.1 × precision_t
        """
        var = prediction_error.var().item() + 1e-8
        new_precision = 1.0 / var
        self._precision = 0.9 * self._precision + 0.1 * new_precision
        self._prediction_errors.append(prediction_error.norm().item())
        if len(self._prediction_errors) > 50:
            self._prediction_errors.pop(0)

    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        FEP: 다음 입력 예측 생성
        
        현재 W와 context로 다음 활성화 패턴 예측
        (W는 세상의 인과 모델을 인코딩)
        """
        if self.wm is None:
            return torch.zeros_like(context)
        # W.T @ context = 입력이 왔을 때 어떤 노드가 활성화될지 예측
        predicted = self.wm.W.T @ context
        predicted = predicted / (predicted.norm() + 1e-8)
        return predicted

    def collect_signal(self) -> RegionSignal:
        """precision, prediction_error 포함 신호 수집"""
        self._ensure_built()

        top = self.hfe.top_hubs(3)
        hub_strength = top[0][1] if top else 0.0
        top_hubs = [(self._nodes[i].name if i < len(self._nodes) else str(i), s)
                    for i, s in top]

        fire_rate = (
            sum(self.wm.recent_fire_rate(n.node_id, 20) for n in self._nodes)
            / max(len(self._nodes), 1)
        )

        # FEP: 예측 오차 기반 CUSUM (Shannon Entropy 대체)
        pred_error = self._compute_prediction_error()
        self._cusum_S = max(0.0, self._cusum_S + pred_error - self._cusum_k)
        overload = self._cusum_S > self._cusum_h

        return RegionSignal(
            region_id        = self.region_name,
            hub_strength     = hub_strength,
            fire_rate        = fire_rate,
            top_hubs         = top_hubs,
            overload         = overload,
            output_vec       = self.wm.W.sum(dim=1).detach().clone(),
            precision        = self._precision,          # ★ FEP
            prediction_error = pred_error,               # ★ FEP
        )

    def _compute_prediction_error(self) -> float:
        """
        FEP: 예측 오차 기반 과부하 신호
        
        기존: Shannon Entropy 발화 집중도 (구조적 과부하)
        수정: 예측 오차 누적 에너지 (기능적 과부하)
        
        수학:
          pred_error = ||actual - predicted||² / precision
          
          예측 오차가 지속적으로 높음 → Region이 현재 입력을
          제대로 모델링하지 못하고 있음 → 진짜 과부하
        
        생물학:
          예측 오차가 해소 안 되면 → 피질 과활성 → 발작/피로
          HTP: 이 상태가 NGE 트리거의 진짜 신호
        """
        if not self._prediction_errors or self._predicted_vec is None:
            return self._entropy_concentration()  # Fallback

        recent_errors = self._prediction_errors[-10:]
        mean_error    = sum(recent_errors) / len(recent_errors)

        # 정밀도로 가중 (신뢰도 낮은 영역의 오차는 덜 심각)
        weighted_error = mean_error / (self._precision + 1e-8)
        return min(weighted_error, 1.0)
```

---

### 수정 3: CoreCells — Precision-weighted Gate

**현재:**
```python
# core_cells.py
# raw score = hub_strength × (1 + fire_rate)
raw: dict[str, float] = {
    sig.region_id: sig.hub_strength * (1.0 + sig.fire_rate)
    for sig in signals
}
```

**수정:**
```python
# core_cells.py

def gate(self, signals: list[RegionSignal],
         top_down: TopDownSignal | None = None) -> GatingMask:
    """
    FEP: Precision-weighted Gating
    
    기존: raw_score = hub_strength × (1 + fire_rate)
          → 활성도가 높은 Region을 게이팅
    
    수정: raw_score = precision_i
          → 예측 오차를 신뢰할 수 있는 Region을 게이팅
    
    수학:
      g_i = σ(β × (π_i + td_bias_i - θ_i))
      π_i = 1 / Var(prediction_error_i)
      
    생물학 (Friston 2010):
      시상은 피질 Region의 정밀도를 조절하는 기관
      정밀도가 높은 Region의 예측 오차가 상위 레벨로 전파됨
      → attention = precision 조절
    """
    if not signals:
        return GatingMask(scores={})

    # FEP: precision 기반 raw score
    raw: dict[str, float] = {
        sig.region_id: sig.precision  # ★ hub_strength → precision
        for sig in signals
    }

    # L1 정규화
    total = sum(raw.values()) or 1.0
    normalized = {rid: v / total for rid, v in raw.items()}

    # Top-down: 목표 관련 Region의 precision 증폭
    td_biases: dict[str, float] = {}
    if top_down and top_down.strength > 0:
        for rid in normalized:
            td_biases[rid] = (
                self._td_weight
                * top_down.biases.get(rid, 0.0)
                * top_down.strength
            )

    # Adaptive θ + Sigmoid
    gated: dict[str, float] = {}
    for rid, score in normalized.items():
        biased_score = score + td_biases.get(rid, 0.0)
        eff_theta    = self.theta + self._theta_bias.get(rid, 0.0)
        gated[rid]   = 1.0 / (1.0 + math.exp(
            -self.beta * (biased_score - eff_theta)
        ))

    return GatingMask(scores=gated)
```

---

### 수정 4: TopDownBias — Softmax 확률 분포 prior

**현재:**
```python
# top_down.py
biases[rid] = len(overlap) / max(len(goal_set), 1)
# → 단순 비율, 확률 분포 아님
```

**수정:**
```python
# top_down.py

import math

def compute(self, goals, regions, step, strength=0.3) -> TopDownSignal:
    """
    FEP: 목표 기반 사전 분포(prior) 생성
    
    기존: Jaccard 비율 → 단순 비율값
    수정: Softmax 정규화 → 확률 분포
    
    수학:
      raw_i  = |goal_tags ∩ region_tags_i| / |goal_tags|
      bias_i = exp(raw_i) / Σ_j exp(raw_j)   (Softmax)
    
    FEP 해석:
      top-down = P(region | goals) = 사전 분포
      Softmax 정규화로 진짜 확률 분포가 됨
      → 시상-피질 루프에서 Bayesian prior로 해석 가능
    
    생물학:
      PFC → 시상 역방향 투영 = 목표 조건부 사전 확률
      특정 목표가 있으면 관련 감각 영역의 precision 사전 상승
    """
    if not goals or not regions:
        return TopDownSignal(biases={}, strength=0.0, step=step)

    goal_set: set[str] = set()
    for g in goals:
        goal_set.update(w.lower() for w in g.replace("_", " ").split())

    raw_biases: dict[str, float] = {}
    for rid, region in regions.items():
        spec      = set(region.specialty.lower().replace("_", " ").split())
        node_tags: set[str] = set()
        for n in getattr(region, "_nodes", []):
            node_tags |= getattr(n.fn, "_htp_tags", set())

        region_tags = spec | node_tags
        overlap     = goal_set & region_tags
        raw_biases[rid] = len(overlap) / max(len(goal_set), 1)

    # ★ Softmax 정규화 (확률 분포화)
    max_val   = max(raw_biases.values()) if raw_biases else 0.0
    exp_vals  = {rid: math.exp(v - max_val) for rid, v in raw_biases.items()}
    total_exp = sum(exp_vals.values()) or 1.0
    biases    = {rid: v / total_exp for rid, v in exp_vals.items()}

    return TopDownSignal(biases=biases, strength=strength, step=step)
```

---

## Phase 3 수정

---

### 수정 5: PFC — Variational Free Energy 결정

**현재:**
```python
# brain_runtime.py
cos_score  = self._cosine_alignment(refined_v)
goal_score = self._goal_alignment(thal_out, regions or {})
score      = (1 - self._goal_alpha) * cos_score + self._goal_alpha * goal_score

if score >= self.inhibition_threshold:
    action = Action(type="execute", ...)
```

**수정:**
```python
# brain_runtime.py

class PFCRuntime(HTPRuntime):
    def __init__(self, config=None):
        super().__init__(config)
        self.working_memory:       deque = deque(maxlen=7)
        self.long_term_goals:      list  = []
        self.inhibition_threshold: float = 0.4
        self._ema_vec:             torch.Tensor | None = None
        self._ema_alpha:           float = 0.7
        self._wm_lambda:           float = 0.4
        self._goal_alpha:          float = 0.4
        self._td_computer:         TopDownBias = TopDownBias()
        # FEP 추가
        self._predicted_vec: torch.Tensor | None = None  # ★
        self._precision:     float = 1.0                  # ★
        self._last_fe:       float = 0.5                  # ★ 마지막 자유 에너지

    def decide(self, thal_out: ThalamusOutput,
               regions: dict = None) -> tuple[Action, TopDownSignal]:
        v = thal_out.state_vec

        # 1. Working Memory Attention
        refined_v = self._wm_attention(v)

        # 2. EMA 업데이트
        self._update_ema(refined_v)

        # 3. WM 저장
        self.working_memory.append(thal_out)

        # 4. ★ FEP: Variational Free Energy 계산 (코사인 유사도 대체)
        fe    = self._variational_free_energy(refined_v)
        score = 1.0 - min(fe, 1.0)  # FE 낮으면 score 높음 → execute

        # 5. 결정
        if score >= self.inhibition_threshold:
            action = Action(
                type   = "execute",
                winner = thal_out.winner,
                reason = f"FE={fe:.3f} score={score:.3f}",
            )
        else:
            action = Action(
                type     = "inhibit",
                winner   = thal_out.winner,
                reason   = f"FE={fe:.3f} score={score:.3f} < {self.inhibition_threshold}",
                redirect = self._find_redirect(thal_out.winner),
            )

        # 6. ★ 다음 상태 예측 업데이트
        self._predicted_vec = refined_v.clone()
        self._last_fe       = fe

        # 7. TopDownSignal 생성
        td_signal = self._td_computer.compute(
            goals    = self.long_term_goals,
            regions  = regions or {},
            step     = len(self.working_memory),
            strength = min(score, 1.0),
        )

        return action, td_signal

    def _variational_free_energy(self, v: torch.Tensor) -> float:
        """
        FEP: Variational Free Energy 근사
        
        F = 예측 오차 에너지 + KL 복잡도
        
        수학:
          prediction_error = v - predicted_v
          energy = ||prediction_error||² / (2 × precision)
          kl     = KL[softmax(v) || softmax(ema_vec)]
          F      = energy + λ × kl
        
        기존 코사인 유사도의 문제:
          cos_sim은 방향만 비교 → 크기(magnitude) 무시
          FE는 오차의 실제 크기를 정밀도로 가중
        
        생물학:
          PFC의 결정 = 자유 에너지 최소화
          낮은 FE → 예측이 잘 맞음 → execute
          높은 FE → 예측이 틀림 → inhibit (재고)
        """
        # 예측 없으면 FE = 중립
        if self._predicted_vec is None or self._predicted_vec.shape != v.shape:
            return 0.5

        # ① 예측 오차 에너지
        error      = v - self._predicted_vec
        energy     = (error ** 2).sum().item() / (2.0 * self._precision + 1e-8)

        # ② KL 복잡도 (현재 상태 vs EMA 사전 분포)
        kl = 0.0
        if self._ema_vec is not None and self._ema_vec.shape == v.shape:
            p = torch.softmax(v, dim=0)
            q = torch.softmax(self._ema_vec, dim=0)
            kl = (p * (p / (q + 1e-8) + 1e-8).log()).sum().item()
            kl = max(0.0, kl)  # KL은 항상 0 이상

        fe = energy + 0.1 * kl
        return min(fe, 2.0)  # 상한선

    # 기존 _cosine_alignment는 보조 지표로 보존
    def _cosine_alignment(self, v: torch.Tensor) -> float:
        """보조 지표 — FEP 전환 전 backward compatibility"""
        if self._ema_vec is None:
            return 1.0
        cos = F.cosine_similarity(v.unsqueeze(0), self._ema_vec.unsqueeze(0)).item()
        return (cos + 1.0) / 2.0
```

---

## Phase 4 수정

---

### 수정 6: PredictiveRegion — 예측 코딩 계층

**현재:**
입력이 오면 처리. 예측 없음.

**수정:**
```python
# region_runtime.py Phase 4 추가

class PredictiveRegion(RegionRuntime):
    """
    FEP: Predictive Coding Region
    
    기존 RegionRuntime: 입력 → 처리 → 출력 (반응 기계)
    수정: 예측 → 오차 → 업데이트 (예측 기계)
    
    처리 순서:
      1. predict(): 다음 입력 예측 생성
      2. run(data): 실제 입력 수신
      3. compute_error(): 예측 - 실제 = prediction error
      4. update_precision(): 오차 분산으로 precision 업데이트
      5. 오차만 상위 레벨(Thalamus)로 전달
    
    생물학:
      피질의 superficial layer → 예측 오차 상향 전달
      피질의 deep layer       → 예측 하향 전달
      시상                    → 정밀도 조절 (attention)
    """

    def __init__(self, region_name, specialty, config=None, gen_config=None):
        super().__init__(region_name, specialty, config, gen_config)
        self._predicted_output: torch.Tensor | None = None

    def predict(self, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        다음 입력 예측 생성
        
        수학:
          predicted = W^T @ last_output   (역방향 투영)
          → W가 세상의 생성 모델을 인코딩
        """
        self._ensure_built()
        if context is None or self.wm.W.numel() == 0:
            n = self.cfg.n_nodes
            return torch.zeros(n, device=self.cfg.device)

        # Deep layer: W.T가 하향 예측 생성
        predicted = self.wm.W.T @ context
        norm = predicted.norm() + 1e-8
        self._predicted_output = predicted / norm
        return self._predicted_output

    def compute_error(self, actual: torch.Tensor) -> torch.Tensor:
        """
        예측 오차 = Prediction Error
        
        수학:
          ε = actual - predicted
          이 오차만 상위 레벨로 전달 (전체 신호가 아니라)
        
        생물학:
          Superficial pyramidal cells가 오차를 상위 영역으로 전달
          Deep pyramidal cells가 예측을 하위 영역으로 전달
        """
        if self._predicted_output is None:
            return actual

        n = min(len(actual), len(self._predicted_output))
        error = actual[:n] - self._predicted_output[:n]
        self.update_precision(error)
        return error

    def run(self, data, entry=None, max_depth=8):
        """
        FEP 확장 run():
          1. 예측 생성 (predict)
          2. 실제 입력으로 처리 (super().run)
          3. 예측 오차 계산
          4. output_vec에 오차만 반영
        """
        # 예측 (이전 출력 기반)
        if self._predicted_output is not None:
            _ = self.predict(self.wm.W.sum(dim=1))

        # 실제 처리
        result = super().run(data, entry, max_depth)

        # 예측 오차 계산
        actual = self.wm.W.sum(dim=1)
        error  = self.compute_error(actual)

        # output_vec을 오차로 교체 (상위 레벨에 오차만 전달)
        # Thalamus는 전체 활성화가 아닌 예측 오차를 받게 됨
        self._last_error = error

        return result

    def collect_signal(self) -> RegionSignal:
        sig = super().collect_signal()
        # output_vec을 예측 오차로 교체
        if hasattr(self, '_last_error') and self._last_error is not None:
            sig.output_vec = self._last_error.detach().clone()
        return sig
```

---

### 수정 7: BrainRuntime — Expected Free Energy 행동 선택

**현재:**
```python
# PFC score >= threshold → execute
# 단순 임계값 비교
```

**수정:**
```python
# brain_runtime.py Phase 4

class PFCRuntime(HTPRuntime):
    def _expected_free_energy(
        self,
        region_name: str,
        thal_out:    ThalamusOutput,
        regions:     dict,
    ) -> float:
        """
        FEP: Expected Free Energy G(π)
        
        수학:
          G(π) = 모호성(ambiguity) + 위험(risk)
          
          모호성 = E_Q[H[P(o|s,π)]]
                 = 이 행동 후 관찰이 얼마나 불확실한가
                 ≈ 1 / precision_i
          
          위험   = KL[Q(o|π) || P(o)]
                 = 이 행동이 선호 결과에서 얼마나 벗어나는가
                 ≈ 1 - goal_alignment
          
          G(π)가 낮은 행동 선택 (정보 획득 + 목표 달성)
        
        생물학:
          PFC의 능동 추론(Active Inference):
          단순히 과거와 비슷한 행동이 아니라
          미래 자유 에너지가 가장 낮은 행동 선택
        """
        region = regions.get(region_name)
        if region is None:
            return float('inf')

        # 모호성: 이 Region의 예측 불확실성
        ambiguity = 1.0 / (getattr(region, '_precision', 1.0) + 1e-8)

        # 위험: 목표에서 이탈 정도
        goal_tags = set(
            w.lower() for g in self.long_term_goals
            for w in g.replace("_", " ").split()
        )
        region_tags: set[str] = set()
        for n in getattr(region, "_nodes", []):
            region_tags |= getattr(n.fn, "_htp_tags", set())

        if goal_tags:
            goal_match = len(goal_tags & region_tags) / len(goal_tags)
        else:
            goal_match = 1.0

        risk = 1.0 - goal_match

        # EFE = 모호성 + 위험 (둘 다 최소화)
        efe = ambiguity + risk
        return efe

    def decide_active_inference(
        self,
        thal_out: ThalamusOutput,
        regions:  dict,
    ) -> tuple[Action, TopDownSignal]:
        """
        FEP: 능동 추론(Active Inference) 행동 선택
        
        기존: score >= threshold → execute (수동적 허용)
        수정: argmin_a G(a) → execute (능동적 최적 선택)
        
        모든 후보 Region에 대해 EFE 계산 후
        가장 낮은 EFE의 Region 선택
        """
        v = thal_out.state_vec
        refined_v = self._wm_attention(v)
        self._update_ema(refined_v)
        self.working_memory.append(thal_out)
        self._predicted_vec = refined_v.clone()

        # ★ 모든 Region에 대해 EFE 계산
        candidates = list(regions.keys()) if regions else [thal_out.winner]
        efe_scores = {
            name: self._expected_free_energy(name, thal_out, regions)
            for name in candidates
        }

        # 최소 EFE Region 선택
        best_region = min(efe_scores, key=efe_scores.get)
        min_efe     = efe_scores[best_region]

        # FE로 execute/inhibit 결정
        fe    = self._variational_free_energy(refined_v)
        score = 1.0 - min(fe, 1.0)

        if score >= self.inhibition_threshold:
            action = Action(
                type   = "execute",
                winner = best_region,        # ★ EFE 최소 Region
                reason = f"EFE={min_efe:.3f} FE={fe:.3f}",
            )
        else:
            action = Action(
                type     = "inhibit",
                winner   = best_region,
                reason   = f"EFE={min_efe:.3f} FE={fe:.3f} < threshold",
                redirect = self._find_redirect(thal_out.winner),
            )

        td_signal = self._td_computer.compute(
            goals    = self.long_term_goals,
            regions  = regions or {},
            step     = len(self.working_memory),
            strength = min(score, 1.0),
        )

        return action, td_signal
```

---

## 전체 수정 흐름도

```
입력 도착
    ↓
┌─────────────────────────────────────────────────────┐
│  PredictiveRegion (Phase 4)                          │
│                                                      │
│  predict() → 예측 생성 (W.T @ context)              │
│      ↓                                               │
│  run(data) → 실제 처리                               │
│      ↓                                               │
│  compute_error() → 예측 오차 ε = actual - predicted  │
│      ↓                                               │
│  update_precision() → π = 1/Var(ε)                  │
│      ↓                                               │
│  collect_signal() → output_vec = ε  (오차만 전달)    │
└─────────────────────────────────────────────────────┘
    ↓ RegionSignal (precision=π, prediction_error=||ε||)
┌─────────────────────────────────────────────────────┐
│  Thalamus — Precision-weighted Gate (Phase 2)        │
│                                                      │
│  CoreCells: g_i = σ(β × (π_i + td_bias_i - θ_i))   │
│             precision 높은 Region을 우선 게이팅       │
│      ↓                                               │
│  MatrixCells: WTA (유사도 기반 국소 억제)             │
│      ↓                                               │
│  JL/PCA 압축 → state_vec (64-dim)                   │
└─────────────────────────────────────────────────────┘
    ↓ ThalamusOutput
┌─────────────────────────────────────────────────────┐
│  PFCRuntime — Variational Free Energy (Phase 3)      │
│                                                      │
│  F = ||v - predicted||² / (2π) + KL[p||q]           │
│      ↓                                               │
│  Active Inference (Phase 4):                         │
│  G(π_i) = 1/precision_i + (1 - goal_match_i)        │
│  best = argmin_i G(π_i)                              │
│      ↓                                               │
│  TopDownSignal = Softmax(goal_biases) = prior        │
└─────────────────────────────────────────────────────┘
    ↓ Action + TopDownSignal
다음 스텝 Thalamus로 피드백
```

---

## 수학 요약

```
Precision (정밀도):
  π_i = 1 / Var(ε_i)
  동적 임계값: threshold_i = base_θ / π_i

Prediction Error:
  ε_i = actual_i - predicted_i
  predicted_i = W_i^T @ context

Variational Free Energy:
  F = ||v - predicted||² / (2π) + λ × KL[p || q]
  score = 1 - min(F, 1.0)

Expected Free Energy (Active Inference):
  G(a_i) = ambiguity_i + risk_i
         = 1/π_i + (1 - goal_match_i)
  best_action = argmin_i G(a_i)

TopDown Prior (Softmax):
  raw_i  = |goal ∩ region_tags_i| / |goal|
  bias_i = exp(raw_i) / Σ_j exp(raw_j)

FEP CUSUM (예측 오차 기반):
  pred_error_weighted = mean(||ε||) / π
  CUSUM_S = max(0, CUSUM_S + pred_error_weighted - k)
  overload = CUSUM_S > h
```

---

## 파일별 수정 위치

```
Phase 2 (즉시):
  region_signal.py      precision, prediction_error 필드 추가
  region_runtime.py     dynamic_threshold, update_precision,
                        _compute_prediction_error 추가
  core_cells.py         gate(): hub_strength → precision 기반
  top_down.py           compute(): Softmax 정규화 추가
  brain_runtime.py      _variational_free_energy() 추가
                        decide(): FE 기반 score로 교체

Phase 3:
  brain_runtime.py      _expected_free_energy() 추가

Phase 4:
  region_runtime.py     PredictiveRegion 클래스 추가
  brain_runtime.py      decide_active_inference() 추가
```

---

*Karl Friston Free Energy Principle 7개 비판 모두 반영*
*참고: Friston 2010 (Free Energy Principle), Friston 2017 (Active Inference),*
*Rao & Ballard 1999 (Predictive Coding), Friston & Kiebel 2009*
