# HTP 전체 시스템 — LeCun 검토 기반 수정 설계

## 비판 요약 및 우선순위

| 우선순위 | 파일 | 문제 | 수정 |
|---------|------|------|------|
| 즉시 (Phase 2) | hub_formation_engine.py | Hebbian variant 불일치 | Oja's Rule 통일 |
| 즉시 (Phase 2) | htp_runtime.py vs hub_formation_engine.py | Hub Detection 혼용 | PageRank 통일 |
| 즉시 (Phase 2) | core_cells.py | homeostatic plasticity 없음 | homeostatic term 추가 |
| 즉시 (Phase 2) | matrix_cells.py | overload 보너스 하드코딩 | 파라미터화 |
| Phase 3 | node_generation_engine.py | Split 기계적 분열 | 기능적 특화 분열 |
| Phase 3 | matrix_cells.py | Global lateral inhibition | 유사도 기반 국소 억제 |
| Phase 4 | activation_engine.py | 문자열 시맨틱 라우팅 | 임베딩 기반 라우팅 |
| Phase 4 | thalamus.py | JL Projection 고정 | Incremental PCA |

---

## 즉시 수정 (Phase 2)

---

### 수정 1: Hebbian → Oja's Rule 통일

**문제:**
```python
# hub_formation_engine.py — BCM-like
delta = cfg.hebbian_lr * co_activation * (1 - W)

# htp_runtime.py — Oja's Rule
oja = torch.outer(y, x) - (y * y).unsqueeze(1) * W
```
두 파일이 서로 다른 Hebbian variant. 어느 게 맞는 규칙인지 모호.

**수정: Oja's Rule로 통일**

수학적 근거:
```
Oja's Rule: Δw_ij = η × y_i × (x_j - y_i × w_ij)
  - y = post-synaptic (fired)
  - x = pre-synaptic (signal)
  - y_i² × w_ij = 정규화 항 (L2 norm 보존)
  - W가 PCA 1st principal component 방향으로 수렴
  - 가중치 폭발 자동 방지 (BCM보다 수학적으로 엄밀)
```

```python
# hub_formation_engine.py 수정
def hebbian_update(self, fired: torch.Tensor, signal: torch.Tensor) -> None:
    """
    Oja's Rule: Δw_ij = η × y_i × (x_j - y_i × w_ij)
    
    기존 co_activation 방식 제거:
      BCM-like (1-W) 포화항 → Oja 정규화항으로 교체
    
    효과:
      - W가 자동으로 L2 정규화 (가중치 폭발 없음)
      - PCA 1st PC 방향으로 수렴 (허브 방향성 명확)
      - htp_runtime.py와 완전 통일
    """
    y   = fired                                           # post [N]
    x   = signal                                          # pre  [N]
    oja = torch.outer(y, x) - (y * y).unsqueeze(1) * self.W
    oja.fill_diagonal_(0)                                 # 자기연결 금지
    self.W += self.cfg.hebbian_lr * oja
    self.W.clamp_(0, 1)

# activate()도 signal을 반환하도록 수정 (hebbian_update에 전달용)
def activate(self, input_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    반환: (fired, input_signal)
    hebbian_update에 pre-synaptic signal 전달 필요
    """
    row_sum = self.W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    W_norm  = self.W / row_sum
    propagated = torch.matmul(W_norm.T, input_signal)
    energy  = propagated * 0.6 + input_signal * 0.4
    fired   = (energy > self.cfg.threshold).float()
    self.fire_count += fired
    return fired, input_signal  # signal도 반환

# step() 수정
def step(self, input_signal: torch.Tensor) -> StepResult:
    self.step_count += 1
    fired, signal = self.activate(input_signal)
    self.hebbian_update(fired, signal)               # signal 전달
    hub_indices = self.detect_hubs()
    pruned = self.prune() if not self.cfg.skip_prune else 0
    return StepResult(...)
```

---

### 수정 2: Hub Detection → PageRank 통일

**문제:**
```python
# hub_formation_engine.py — In-strength (단순 합)
in_strength = W.sum(dim=0)
self.is_hub = in_strength > cfg.hub_threshold

# htp_runtime.py — PageRank (중요도 전파)
def pagerank(self, alpha=0.85): ...  # Power Iteration
```

**생물학적 근거:**
해마 허브 뉴런은 단순히 연결이 많은 게 아니라
중요한 뉴런들과 연결된 것이 허브.
→ PageRank가 생물학적으로 더 정확.

**수정: PageRank로 통일 + threshold 자동 계산**

```python
# hub_formation_engine.py 수정
def detect_hubs(self) -> torch.Tensor:
    """
    PageRank 기반 허브 감지 (In-strength 제거)
    
    수학:
      r = α × W_col_norm^T × r + (1-α)/N
      hub = r > mean(r) + std(r)  (통계적 임계값)
    
    기존 hub_threshold 파라미터 → 통계적 z-score로 대체:
      허브 = PageRank 점수가 평균 + k×std 이상인 노드
      (k=1.0 기본값 — 상위 ~16%)
    """
    pr = self.pagerank()
    
    # 통계적 임계값: 평균 + 1σ 이상 = 허브
    mu  = pr.mean()
    sig = pr.std() + 1e-8
    self.is_hub = pr > (mu + self.cfg.hub_z * sig)
    
    return self.is_hub.nonzero(as_tuple=True)[0]

def pagerank(self, alpha: float = 0.85,
             tol: float = 1e-5, max_iter: int = 30) -> torch.Tensor:
    """Power Iteration PageRank (htp_runtime.py에서 이동)"""
    W   = self.W
    N   = W.shape[0]
    col = W.sum(dim=0, keepdim=True).clamp(min=1e-8)
    Wc  = W / col
    r   = torch.ones(N, device=W.device) / N
    tp  = (1 - alpha) / N
    for _ in range(max_iter):
        r_new = alpha * (Wc.T @ r) + tp
        if (r_new - r).abs().max() < tol:
            break
        r = r_new
    return r

# HTPConfig에 추가
@dataclass
class HTPConfig:
    ...
    hub_z: float = 1.0  # PageRank z-score 임계값 (기존 hub_threshold 대체)
    # hub_threshold 제거
```

---

### 수정 3: CoreCells — Homeostatic Plasticity 추가

**문제:**
```python
# 현재: 단조 감소 — 자주 이기면 임계값이 계속 낮아짐
bias -= eta * win_history[rid]
theta_bias[rid] = max(-0.2, min(0.2, bias))
```
winner-takes-all 붕괴 위험.

**수학적 근거:**
```
Homeostatic Plasticity (Turrigiano 2008):
  목표 발화율(target_rate)에서 벗어나면 음성 피드백
  error = win_rate - target_rate
  Δθ = -η × error
  → 승률이 너무 높으면 θ 올라감 (억제)
  → 승률이 너무 낮으면 θ 내려감 (촉진)
```

```python
# core_cells.py 수정
def update(self, winner_id: str, all_ids: list[str]):
    """
    Hebbian + Homeostatic Plasticity
    
    기존: 승리율에 비례해 단조 감소 → 붕괴 위험
    수정: target_rate 중심 음성 피드백
    
    수학:
      win_history[i] = 0.1 × win_t + 0.9 × win_history[i]  (EMA)
      error[i]       = win_history[i] - target_rate
      theta_bias[i] -= eta × error[i]
      clamp: theta_bias ∈ [-0.3, 0.3]
    
    효과:
      승률 > target → θ 올라감 → 게이팅 어려워짐 (억제)
      승률 < target → θ 내려감 → 게이팅 쉬워짐  (촉진)
      장기적으로 모든 Region이 target_rate 근처로 수렴
    """
    for rid in all_ids:
        win  = 1.0 if rid == winner_id else 0.0
        prev = self._win_history.get(rid, 0.0)
        self._win_history[rid] = 0.1 * win + 0.9 * prev

    for rid in all_ids:
        error = self._win_history.get(rid, 0.0) - self._target_rate
        bias  = self._theta_bias.get(rid, 0.0)
        bias -= self._eta * error                    # 음성 피드백
        self._theta_bias[rid] = max(-0.3, min(0.3, bias))

def __init__(self,
             beta:        float = 5.0,
             theta:       float = 0.3,
             eta:         float = 0.05,
             td_weight:   float = 0.3,
             target_rate: float = 0.3):   # 추가: 이상적 승률
    ...
    self._target_rate = target_rate
```

---

### 수정 4: MatrixCells — 파라미터화 + 국소 억제 (즉시)

**문제:**
```python
# 하드코딩된 overload 보너스
raw = gating.scores.get(...) + (0.2 if sig.overload else 0.0)
```

**즉시 수정: 파라미터화**

```python
# matrix_cells.py 수정
def __init__(self,
             temperature:    float = 1.0,
             lateral_w:      float = 0.15,
             lateral_iter:   int   = 3,
             overload_bonus: float = 0.2):   # 추가: 파라미터화
    ...
    self.overload_bonus = overload_bonus

def compete(self, signals, gating) -> CompetitionResult:
    raw = torch.tensor([
        gating.scores.get(sig.region_id, 0.0)
        + (self.overload_bonus if sig.overload else 0.0)  # 파라미터 사용
        for sig in signals
    ], dtype=torch.float32)
    ...
```

---

## Phase 3 수정

---

### 수정 5: MatrixCells — Global → 유사도 기반 국소 Lateral Inhibition

**문제:**
```python
# 현재: Global Inhibition — 모든 Region이 동등하게 억제
inhibition = lateral_w * (total - s)  # 자신 제외 전체 합
```

**생물학적 근거:**
실제 시상 Matrix cells의 lateral inhibition은
유사한 기능 영역끼리 더 강하게 억제
(같은 감각 모달리티끼리 경쟁)

**수정: Region specialty 유사도 기반 국소 억제**

```python
# matrix_cells.py Phase 3 수정

def compete(self, signals: list[RegionSignal],
            gating: GatingMask) -> CompetitionResult:
    """
    유사도 기반 국소 Lateral Inhibition
    
    수학:
      sim_ij = cosine(specialty_i, specialty_j)  -- Region 유사도
      inhibition_i = lateral_w × Σ_j sim_ij × s_j  (유사할수록 더 억제)
      s_i(t+1) = ReLU(s_i(t) - inhibition_i)
    
    효과:
      - 같은 모달리티(예: language vs language') 끼리 강하게 경쟁
      - 다른 모달리티(예: language vs vision) 는 약하게 경쟁
      - 생물학: 같은 감각 피질 내 컬럼 간 억제와 동일
    """
    region_ids = [sig.region_id for sig in signals]
    N = len(signals)

    raw = torch.tensor([
        gating.scores.get(sig.region_id, 0.0)
        + (self.overload_bonus if sig.overload else 0.0)
        for sig in signals
    ], dtype=torch.float32)

    # Region 유사도 행렬 계산 (specialty 문자열 → 간단 임베딩)
    sim_matrix = self._compute_similarity(signals)  # [N, N]

    # 유사도 기반 국소 Lateral Inhibition
    s = raw.clone()
    for _ in range(self.lateral_iter):
        inhibition = self.lateral_w * (sim_matrix @ s)  # 유사한 것만 억제
        inhibition.fill_diagonal_(0)                     # 자기 억제 제거
        s = torch.relu(s - inhibition.sum(dim=1))

    probs    = torch.softmax(s / max(self.temperature, 1e-6), dim=0)
    winner_idx   = int(probs.argmax().item())
    winner_id    = region_ids[winner_idx]
    winner_score = float(probs[winner_idx])

    suppression = {
        region_ids[i]: max(0.0, winner_score - float(probs[i]))
        for i in range(N) if i != winner_idx
    }
    return CompetitionResult(
        winner_id=winner_id,
        winner_score=winner_score,
        suppression_map=suppression,
        all_scores={rid: float(p) for rid, p in zip(region_ids, probs.tolist())},
    )

def _compute_similarity(self, signals: list[RegionSignal]) -> torch.Tensor:
    """
    Region specialty 간 문자열 Jaccard 유사도 행렬
    Phase 4에서 임베딩 기반으로 업그레이드 예정
    """
    N   = len(signals)
    sim = torch.eye(N)  # 자기 자신은 1.0
    for i in range(N):
        for j in range(i+1, N):
            words_i = set(signals[i].region_id.lower().split("_"))
            words_j = set(signals[j].region_id.lower().split("_"))
            jaccard  = len(words_i & words_j) / max(len(words_i | words_j), 1)
            sim[i][j] = sim[j][i] = jaccard
    return sim
```

---

### 수정 6: NodeGenerationEngine — 기능적 특화 분열

**문제:**
```python
# 현재: 강도 기반 기계적 분열
in_weights.sort(key=lambda x: -x[1])
child_a → in_weights[:mid_in]   # 강한 연결
child_b → in_weights[mid_in:]   # 약한 연결
```
child_b가 처음부터 불리. 기능적 다양성 없음.

**생물학적 근거:**
피질 컬럼 분화는 입력 패턴의 유사성으로 특화
(시각 컬럼 → 방향 선택성으로 분화)

**수정: 입력 연결 패턴의 유사성으로 특화 분열**

```python
# node_generation_engine.py Phase 3 수정

def _do_split(self, parent: Node, step: int) -> tuple:
    """
    기능적 특화 분열 (LeCun 수정)
    
    기존: 연결 강도 기준 절반 상속 → 불균등
    수정: 입력 연결 패턴 유사성으로 클러스터링
    
    수학:
      각 입력 노드 u의 연결 패턴: W[u, :] (출력 패턴)
      유사한 입력 패턴 → child_a
      다른 입력 패턴  → child_b
      (2-means clustering on input patterns)
    
    생물학:
      피질 컬럼 분화: 유사한 자극에 반응하는 뉴런끼리 군집
      → 기능적으로 특화된 두 서브컬럼 형성
    """
    N   = self.wm.n
    W   = self.wm.W
    pid = parent.node_id

    in_nodes = [u for u in range(N) if float(W[u][pid]) > 0]
    if len(in_nodes) < 2:
        # 연결이 너무 적으면 기존 방식 fallback
        return self._do_split_legacy(parent, step)

    # 입력 노드들의 출력 패턴 수집
    patterns = torch.stack([W[u].clone() for u in in_nodes])  # [k, N]

    # 2-means: 랜덤 초기화 후 수렴
    c0 = patterns[0]
    c1 = patterns[-1]
    for _ in range(10):  # 최대 10 iteration
        dists_0 = ((patterns - c0) ** 2).sum(dim=1)
        dists_1 = ((patterns - c1) ** 2).sum(dim=1)
        labels  = (dists_1 < dists_0).long()  # 0 or 1
        new_c0  = patterns[labels == 0].mean(dim=0) if (labels == 0).any() else c0
        new_c1  = patterns[labels == 1].mean(dim=0) if (labels == 1).any() else c1
        if (new_c0 - c0).norm() < 1e-4 and (new_c1 - c1).norm() < 1e-4:
            break
        c0, c1 = new_c0, new_c1

    group_a = [in_nodes[i] for i in range(len(in_nodes)) if labels[i] == 0]
    group_b = [in_nodes[i] for i in range(len(in_nodes)) if labels[i] == 1]

    # 균등 분배 보장 (한쪽이 비면 절반씩)
    if not group_a or not group_b:
        mid = len(in_nodes) // 2
        group_a, group_b = in_nodes[:mid], in_nodes[mid:]

    out_nodes = [(v, float(W[pid][v])) for v in range(N) if float(W[pid][v]) > 0]
    mid_out   = max(1, len(out_nodes) // 2)

    child_a = self._create_node(
        f"{parent.name}_a", parent.fn,
        in_edges  = [(u, float(W[u][pid])) for u in group_a],
        out_edges = out_nodes[:mid_out],
    )
    child_b = self._create_node(
        f"{parent.name}_b", parent.fn,
        in_edges  = [(u, float(W[u][pid])) for u in group_b],
        out_edges = out_nodes[mid_out:],
    )

    # 두 자식 간 약한 양방향 연결 (cortico-cortical)
    self.wm.set(child_a.node_id, child_b.node_id, 0.05)
    self.wm.set(child_b.node_id, child_a.node_id, 0.05)

    # 부모 점진적 약화
    self.wm.W[pid] *= 0.5
    self.wm.W[:, pid] *= 0.5

    self._immature.add(child_a.node_id)
    self._immature.add(child_b.node_id)

    print(f"  [GEN] SPLIT(functional) {parent.name} -> {child_a.name}({len(group_a)} in) + {child_b.name}({len(group_b)} in)")
    return child_a, child_b
```

---

## Phase 4 수정

---

### 수정 7: ActivationEngine — 임베딩 기반 시맨틱 라우팅

**문제:**
```python
# 현재: 문자열 교집합
overlap = ntags & (kws | {label})
boost   = min(0.4 + 0.20 * len(overlap), 0.90)
```
"success"와 "ok"가 같은 의미임을 모름.

**수정: LLMNode 임베딩 캐시 활용**

```python
# activation_engine.py Phase 4 수정

class ActivationEngine:
    def __init__(self, wm, hfe, cfg):
        ...
        self._embed_cache: dict[str, torch.Tensor] = {}  # 임베딩 캐시

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        텍스트 → 임베딩 벡터
        Phase 4: Anthropic Embedding API 사용
        Fallback: TF-IDF 기반 sparse vector
        """
        if text in self._embed_cache:
            return self._embed_cache[text]

        try:
            import anthropic
            client  = anthropic.Anthropic()
            resp    = client.embeddings.create(
                model = "voyage-3",
                input = [text],
            )
            vec = torch.tensor(resp.embeddings[0].embedding)
        except Exception:
            # Fallback: 문자 기반 해시 벡터 (API 없을 때)
            vec = self._hash_embed(text)

        self._embed_cache[text] = vec
        return vec

    def _make_signal(self, data, prev_ids, visited):
        N   = len(self._nodes)
        sig = torch.zeros(N, device=self.cfg.device)

        for uid in prev_ids:
            sig[uid] = 1.0

        # 데이터 임베딩
        data_str = str(data) if not isinstance(data, str) else data
        data_vec = self._get_embedding(data_str[:200])

        for n in self._nodes:
            ntags = getattr(n.fn, "_htp_tags", set())
            if not ntags:
                continue

            # 태그 임베딩과 데이터 임베딩 코사인 유사도
            tag_str = " ".join(ntags)
            tag_vec = self._get_embedding(tag_str)

            import torch.nn.functional as F
            sim  = F.cosine_similarity(
                data_vec.unsqueeze(0), tag_vec.unsqueeze(0)
            ).item()
            sim  = (sim + 1.0) / 2.0  # [-1,1] → [0,1]

            if sim > 0.3:  # 유사도 임계값
                boost = min(0.3 + 0.6 * sim, 0.90)
                if sig[n.node_id].item() < boost:
                    sig[n.node_id] = boost

        return sig.clamp(0.0, 1.0)

    def _hash_embed(self, text: str, dim: int = 64) -> torch.Tensor:
        """API 없을 때 Fallback — 문자 기반 해시 벡터"""
        vec = torch.zeros(dim)
        for i, c in enumerate(text):
            vec[ord(c) % dim] += 1.0 / (i + 1)
        norm = vec.norm() + 1e-8
        return vec / norm
```

---

### 수정 8: Thalamus — JL Projection → Incremental PCA

**문제:**
```python
# 현재: 초기화 후 고정
if n not in self._proj:
    Phi = torch.randn(k, n)
    self._proj[n] = Phi  # 고정!
```
네트워크가 학습하면서 W가 변하는데 projection은 고정.

**수정: Incremental PCA로 학습 가능한 압축**

```python
# thalamus.py Phase 4 수정

class Thalamus:
    def __init__(self, ...):
        ...
        self._pca: dict[int, IncrementalPCA] = {}

    def _jl_compress(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Incremental PCA 압축 (JL Random Projection 대체)
        
        기존: 고정 랜덤 행렬 Φ ~ N(0, 1/k)
        수정: 온라인 PCA — 입력 분포에 적응
        
        수학:
          PCA: z = U^T × (x - μ)
          U = top-k eigenvectors of covariance matrix
          Incremental: 새 샘플마다 U 점진적 업데이트
        
        초기 (샘플 < k): JL Projection fallback
        이후: Incremental PCA
        """
        k = self.compress_dim
        n = len(vec)

        if n not in self._pca:
            self._pca[n] = IncrementalPCA(n_components=k)

        pca = self._pca[n]
        return pca.transform(vec)


class IncrementalPCA:
    """
    단순 Incremental PCA 구현
    (sklearn 의존성 없이 순수 torch)
    """
    def __init__(self, n_components: int, lr: float = 0.01):
        self.k   = n_components
        self.lr  = lr
        self.U   = None   # [n, k] — principal components
        self.mu  = None   # [n]    — running mean
        self.t   = 0      # 샘플 수

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        self.t += 1

        # 평균 업데이트 (온라인)
        if self.mu is None:
            self.mu = x.clone()
        else:
            self.mu = self.mu + (x - self.mu) / self.t

        x_centered = x - self.mu

        # 초기: JL Fallback (U 없을 때)
        if self.U is None:
            n = len(x)
            if self.t >= self.k:
                # 충분한 샘플 → PCA 초기화
                self.U = torch.randn(n, self.k)
                self.U, _ = torch.linalg.qr(self.U)
            else:
                # JL Fallback
                Phi = torch.randn(self.k, n) / (self.k ** 0.5)
                return Phi @ x_centered

        # Incremental PCA: Oja's subspace learning
        # Δu_i = η × (x - u_i × u_i^T × x)
        proj = self.U.T @ x_centered          # [k]
        recon = self.U @ proj                  # [n]
        error = x_centered - recon            # [n]
        self.U = self.U + self.lr * torch.outer(error, proj)

        # 정규직교화 (QR)
        self.U, _ = torch.linalg.qr(self.U)

        return proj  # [k] — 압축 벡터
```

---

## 전체 수정 흐름도

```
Phase 2 (즉시):
  hub_formation_engine.py
    hebbian_update()     BCM-like → Oja's Rule
    detect_hubs()        In-strength → PageRank
    activate()           signal 반환 추가

  core_cells.py
    update()             단조 감소 → Homeostatic Plasticity
    __init__()           target_rate 파라미터 추가

  matrix_cells.py
    __init__()           overload_bonus 파라미터 추가
    compete()            하드코딩 제거

  htp_runtime.py
    HTPConfig            hub_threshold 제거, hub_z 추가

Phase 3:
  matrix_cells.py
    compete()            Global → 유사도 기반 국소 Lateral Inhibition
    _compute_similarity() 추가

  node_generation_engine.py
    _do_split()          강도 기반 → 기능적 특화 (2-means)
    _do_split_legacy()   기존 방식 fallback으로 보존

Phase 4:
  activation_engine.py
    _make_signal()       문자열 교집합 → 임베딩 코사인 유사도
    _get_embedding()     추가 (캐시 포함)
    _hash_embed()        Fallback 추가

  thalamus.py
    _jl_compress()       고정 JL → Incremental PCA
    IncrementalPCA       새 클래스 추가
    compress_dim         8 → 64
```

---

## 수학 요약

```
Oja's Rule:
  Δw_ij = η × y_i × (x_j - y_i × w_ij)
  → W가 covariance 주성분 방향으로 수렴

PageRank Hub:
  r = α × W_col_norm^T × r + (1-α)/N
  hub = r > μ(r) + hub_z × σ(r)

Homeostatic Plasticity:
  error_i = win_rate_i - target_rate
  Δθ_i   = -η × error_i

유사도 기반 억제:
  sim_ij = cosine(embed(specialty_i), embed(specialty_j))
  inhib_i = lateral_w × Σ_j sim_ij × s_j

기능적 특화 분열:
  patterns_u = W[u, :]  for u in in_nodes
  labels = 2-means(patterns)
  child_a ← group_0, child_b ← group_1

Incremental PCA (Oja's Subspace):
  proj  = U^T × (x - μ)
  error = (x - μ) - U × proj
  ΔU    = η × outer(error, proj)
  U     = QR(U + ΔU).Q
```

---

*LeCun 검토 8개 항목 전체 반영*
*참고: Oja 1982, Turrigiano 2008, BCM Theory,*
*PageRank (Page et al. 1999), Incremental PCA (Warmuth & Kuzmin 2008)*
