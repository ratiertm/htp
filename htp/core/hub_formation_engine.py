"""
Hub Formation Engine
허브 토폴로지 프로그래밍 (HTP) 런타임의 핵심 엔진

세 가지 원리:
  1. Hebbian Learning  — 함께 발화하는 노드는 함께 연결된다
  2. Hub Detection     — 연결 강도가 임계값을 넘으면 허브로 승격
  3. Synaptic Pruning  — 안 쓰는 연결은 시간 감쇠로 소멸
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

@dataclass
class HTPConfig:
    n_nodes: int = 64              # 노드 수
    threshold: float = 0.5        # 노드 발화 임계값
    hub_threshold: float = 3.0    # 허브 승격 임계값 (연결 강도 합계)
    hebbian_lr: float = 0.1       # 헤비안 학습률
    decay_rate: float = 0.005     # 연결 시간 감쇠율
    prune_threshold: float = 0.02 # 이 이하면 연결 제거
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    skip_prune: bool = False      # True → step()에서 prune() 생략 (PruningEngine 위임)


# ─────────────────────────────────────────────
# Hub Formation Engine
# ─────────────────────────────────────────────

class HubFormationEngine:
    """
    핵심 엔진.
    매 스텝마다:
      activate → hebbian_update → hub_detect → prune
    """

    def __init__(self, config: HTPConfig):
        self.cfg = config
        self.dev = config.device

        # 연결 가중치 행렬 [n x n], 희소 구조로 시작
        self.W = torch.zeros(
            config.n_nodes, config.n_nodes, device=self.dev
        )
        # 초기 연결: 랜덤하게 약한 연결 부여 (밀도 20%)
        mask = torch.rand_like(self.W) < 0.2
        self.W[mask] = torch.rand(mask.sum(), device=self.dev) * 0.3

        # 자기 자신으로의 연결은 없음
        self.W.fill_diagonal_(0)

        # 각 노드의 누적 발화 횟수 (허브 감지용)
        self.fire_count = torch.zeros(config.n_nodes, device=self.dev)

        # 허브 노드 마스크
        self.is_hub = torch.zeros(
            config.n_nodes, dtype=torch.bool, device=self.dev
        )

        self.step_count = 0

    # ── 1. Activation ──────────────────────────────

    def activate(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        입력 신호를 받아 네트워크 전체로 에너지를 전파.
        E[v] = Σ w[u→v] × E[u]  (row-normalized W로 포화 방지)
        임계값 넘은 노드만 발화 (랑비에 결절 원리)
        """
        # 행 정규화: 각 노드의 출력 가중치 합을 1로 제한
        row_sum = self.W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = self.W / row_sum

        # 에너지 전파
        propagated = torch.matmul(W_norm.T, input_signal)

        # 직접 입력 + 전파 에너지 합산 후 임계값 적용
        energy = propagated * 0.6 + input_signal * 0.4
        fired = (energy > self.cfg.threshold).float()

        # 발화 횟수 누적
        self.fire_count += fired

        return fired

    # ── 2. Hebbian Update ──────────────────────────

    def hebbian_update(self, fired: torch.Tensor) -> None:
        """
        함께 발화한 노드 쌍의 연결만 강화.
        Δw[u→v] = α × fire(u) × fire(v)
        단, 이미 강한 연결은 천천히 강화 (포화 방지)
        """
        co_activation = torch.outer(fired, fired)
        co_activation.fill_diagonal_(0)

        # 이미 강한 연결은 덜 강화 (1 - W 비례)
        delta = self.cfg.hebbian_lr * co_activation * (1 - self.W)
        self.W += delta
        self.W.clamp_(0, 1)

    # ── 3. Hub Detection ───────────────────────────

    def detect_hubs(self) -> torch.Tensor:
        """
        각 노드의 들어오는 연결 강도 합계가
        hub_threshold를 넘으면 허브로 승격.
        반환: 허브 노드 인덱스 리스트
        """
        in_strength = self.W.sum(dim=0)  # 각 노드로 들어오는 연결 합
        self.is_hub = in_strength > self.cfg.hub_threshold
        return self.is_hub.nonzero(as_tuple=True)[0]

    # ── 4. Pruning ─────────────────────────────────

    def prune(self) -> int:
        """
        시간 감쇠 적용 후 약한 연결 제거.
        뇌의 시냅스 가지치기와 동일한 원리.
        반환: 제거된 연결 수
        """
        # 시간 감쇠 (매 스텝 조금씩 약해짐)
        self.W *= (1 - self.cfg.decay_rate)

        # 임계값 이하 연결 제거
        weak = self.W < self.cfg.prune_threshold
        pruned_count = int(weak.sum().item())
        self.W[weak] = 0

        return pruned_count

    # ── 전체 스텝 ──────────────────────────────────

    def step(self, input_signal: torch.Tensor) -> "StepResult":
        """
        한 스텝: activate → hebbian → hub_detect → prune
        """
        self.step_count += 1

        fired       = self.activate(input_signal)
        self.hebbian_update(fired)
        hub_indices = self.detect_hubs()
        pruned      = self.prune() if not self.cfg.skip_prune else 0

        return StepResult(
            step        = self.step_count,
            fired       = fired,
            hub_indices = hub_indices,
            n_hubs      = len(hub_indices),
            n_edges     = int((self.W > 0).sum().item()),
            pruned      = pruned,
            avg_weight  = float(self.W[self.W > 0].mean().item())
                          if (self.W > 0).any() else 0.0,
        )

    # ── 유틸 ───────────────────────────────────────

    def top_hubs(self, k: int = 5) -> list:
        """연결 강도 기준 상위 k개 허브 노드 반환"""
        in_strength = self.W.sum(dim=0)
        values, indices = torch.topk(in_strength, k=min(k, self.cfg.n_nodes))
        return [(int(idx), float(val)) for idx, val in zip(indices, values)]

    def edge_count(self) -> int:
        return int((self.W > 0).sum().item())


# ─────────────────────────────────────────────
# 결과 컨테이너
# ─────────────────────────────────────────────

@dataclass
class StepResult:
    step:        int
    fired:       torch.Tensor
    hub_indices: torch.Tensor
    n_hubs:      int
    n_edges:     int
    pruned:      int
    avg_weight:  float

    def __repr__(self):
        hub_list = self.hub_indices.tolist()[:5]
        return (
            f"Step {self.step:>4} | "
            f"fired={int(self.fired.sum()):>3} | "
            f"hubs={self.n_hubs:>2} {hub_list} | "
            f"edges={self.n_edges:>5} | "
            f"pruned={self.pruned:>4} | "
            f"avg_w={self.avg_weight:.4f}"
        )


# ─────────────────────────────────────────────
# 간단한 데모
# ─────────────────────────────────────────────

def demo():
    print("=" * 70)
    print("  Hub Formation Engine — Demo")
    print("=" * 70)

    cfg = HTPConfig(
        n_nodes         = 32,
        threshold       = 0.45,
        hub_threshold   = 4.0,
        hebbian_lr      = 0.08,
        decay_rate      = 0.008,
        prune_threshold = 0.03,
    )
    engine = HubFormationEngine(cfg)

    print(f"\n초기 상태")
    print(f"  노드 수   : {cfg.n_nodes}")
    print(f"  초기 엣지 : {engine.edge_count()}")
    print(f"  디바이스  : {cfg.device}")
    print()

    # 반복 패턴: 노드 0~7이 자주 함께 활성화 → 허브 형성 유도
    for i in range(1, 101):
        signal = torch.zeros(cfg.n_nodes, device=cfg.device)

        if i % 3 == 0:
            # 패턴 A: 0~7 강하게 활성화
            signal[:8] = torch.rand(8, device=cfg.device) * 0.8 + 0.4
        elif i % 3 == 1:
            # 패턴 B: 0~3 + 16~19 활성화
            signal[:4]    = torch.rand(4, device=cfg.device) * 0.7 + 0.3
            signal[16:20] = torch.rand(4, device=cfg.device) * 0.7 + 0.3
        else:
            # 노이즈
            signal = torch.rand(cfg.n_nodes, device=cfg.device) * 0.3

        result = engine.step(signal)

        if i in (1, 10, 30, 60, 100):
            print(result)

    print()
    print("─" * 70)
    print("  최종 상위 허브 노드 (연결 강도 기준)")
    print("─" * 70)
    for rank, (node_id, strength) in enumerate(engine.top_hubs(5), 1):
        bar = "█" * int(strength * 10)
        hub_mark = " ★ HUB" if engine.is_hub[node_id] else ""
        print(f"  #{rank}  Node {node_id:>2}  strength={strength:.3f}  {bar}{hub_mark}")

    print()
    print(f"  총 엣지 수 : {engine.edge_count()}")
    print(f"  허브 노드  : {engine.is_hub.sum().item()}개")
    print("=" * 70)


if __name__ == "__main__":
    demo()
