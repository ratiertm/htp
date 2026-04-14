# HTP Multimodal Extension — V-JEPA 방식 설계

## 핵심 원리

```
V-JEPA 핵심 통찰:
  "픽셀을 예측하지 말고, 잠재 공간에서 예측하라"
  
HTP 적용:
  각 모달리티 → 모달리티별 행렬 → 공유 잠재 공간 → Thalamus 통합
  
르쿤 + Friston 결합:
  각 Region = V-JEPA Encoder (모달리티 → 잠재 행렬)
  Thalamus  = Fusion Tokens  (잠재 행렬 → 통합 벡터)
  PFC       = Predictor      (미래 상태 예측 + EFE 결정)
```

---

## 1. 아키텍처 전체 흐름

```
┌──────────────────────────────────────────────────────────────┐
│                    멀티모달 입력 계층                          │
│                                                              │
│  LiDAR    Camera    Audio    Text    IMU    Tactile          │
│    ↓         ↓        ↓       ↓       ↓       ↓             │
│  [N×4]   [H×W×3]  [F×T]  [L×d]   [6×T]   [P×T]           │
│  행렬     tubelet   스펙트  토큰    가속도   압력              │
└──────────────────────────────────────────────────────────────┘
          ↓ 모달리티별 Encoder (RegionRuntime)
┌──────────────────────────────────────────────────────────────┐
│                  Modal Region 계층                            │
│                                                              │
│  LiDARRegion  CameraRegion  AudioRegion  TextRegion         │
│  [N×4]→[d]   [P×d]→[P×d]  [F×T]→[d]   [L×d]→[d]         │
│                                                              │
│  각 Region이 자신의 모달리티를 잠재 벡터/행렬로 변환          │
│  + 예측 생성 (PredictiveRegion)                              │
│  + precision 계산 (Friston)                                  │
└──────────────────────────────────────────────────────────────┘
          ↓ RegionSignal (modal_matrix 포함)
┌──────────────────────────────────────────────────────────────┐
│                  Thalamus — Fusion Token 계층                 │
│                                                              │
│  Le MuMo JEPA 방식:                                         │
│  1. 각 Region의 modal_matrix 수집                            │
│  2. Cross-modal Attention (Fusion Tokens)                    │
│  3. Pruning: 모달리티별 토큰 제거 → 공유 잠재 벡터           │
│  4. 64-dim state_vec 생성 (Friston precision 가중)           │
└──────────────────────────────────────────────────────────────┘
          ↓ ThalamusOutput (state_vec + modal_weights)
┌──────────────────────────────────────────────────────────────┐
│                  PFC — World Model Predictor                  │
│                                                              │
│  V-JEPA Predictor 방식:                                      │
│  1. Variational Free Energy 계산 (Friston)                   │
│  2. 다음 상태 예측 (predict_next)                             │
│  3. Expected Free Energy로 행동 선택 (Active Inference)      │
│  4. TopDownSignal → 각 Region으로 피드백                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 모달리티별 행렬 인코딩

### 2-1. ModalEncoder 기반 클래스

```python
# htp/multimodal/modal_encoder.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ModalMatrix:
    """
    모달리티별 잠재 행렬
    
    V-JEPA tubelet 개념의 일반화:
      모든 모달리티를 [n_patches × d] 형태로 통일
    """
    modality:   str           # "lidar" | "camera" | "audio" | "text" | "imu"
    matrix:     torch.Tensor  # [n_patches × embed_dim]
    positions:  torch.Tensor  # [n_patches × pos_dim] 위치 인코딩
    mask:       torch.Tensor  # [n_patches] bool — 유효한 패치
    timestamp:  float         # 수집 시각


class ModalEncoder(ABC):
    """
    모달리티 → ModalMatrix 변환기 기반 클래스
    
    V-JEPA: 각 모달리티를 공통 잠재 공간으로 인코딩
    Friston: 인코딩 과정에서 precision 계산
    """

    def __init__(self, embed_dim: int = 64, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.device    = device
        self._precision: float = 1.0

    @abstractmethod
    def encode(self, raw_input) -> ModalMatrix:
        """raw 입력 → ModalMatrix"""
        pass

    def update_precision(self, prediction_error: torch.Tensor):
        """Friston: 예측 오차로 precision 업데이트"""
        var = prediction_error.var().item() + 1e-8
        new_p = 1.0 / var
        self._precision = 0.9 * self._precision + 0.1 * new_p

    @property
    def precision(self) -> float:
        return self._precision
```

---

### 2-2. LiDAR Encoder

```python
# htp/multimodal/encoders/lidar_encoder.py

import torch
import torch.nn.functional as F
from ..modal_encoder import ModalEncoder, ModalMatrix


class LiDAREncoder(ModalEncoder):
    """
    LiDAR 포인트 클라우드 → ModalMatrix
    
    입력: [N × 4] (x, y, z, intensity)
    출력: [n_voxels × embed_dim]
    
    V-JEPA tubelet 대응:
      3D 공간을 voxel grid로 분할 = tubelet 분할과 동일
      각 voxel = 하나의 patch
    
    처리:
      1. Voxel Grid 분할 (공간 분할)
      2. 각 voxel의 포인트 집계 (PointNet-lite)
      3. 위치 인코딩 (3D-RoPE 단순화 버전)
    """

    def __init__(self,
                 embed_dim:   int   = 64,
                 voxel_size:  float = 0.5,   # 미터 단위
                 max_voxels:  int   = 128,
                 device:      str   = "cpu"):
        super().__init__(embed_dim, device)
        self.voxel_size = voxel_size
        self.max_voxels = max_voxels

        # 간단한 선형 인코더 (PointNet-lite)
        self.point_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, embed_dim),
        ).to(device)

    def encode(self, points: torch.Tensor) -> ModalMatrix:
        """
        points: [N × 4] (x, y, z, intensity)
        """
        if points.numel() == 0:
            empty = torch.zeros(1, self.embed_dim, device=self.device)
            pos   = torch.zeros(1, 3, device=self.device)
            mask  = torch.zeros(1, dtype=torch.bool, device=self.device)
            return ModalMatrix("lidar", empty, pos, mask, 0.0)

        # Voxel Grid 분할
        xyz = points[:, :3]
        voxel_idx = (xyz / self.voxel_size).long()

        # 고유 voxel 찾기
        unique_voxels, inverse = torch.unique(
            voxel_idx, dim=0, return_inverse=True
        )
        n_voxels = min(len(unique_voxels), self.max_voxels)
        unique_voxels = unique_voxels[:n_voxels]

        # 각 voxel의 포인트 집계 → 임베딩
        embeddings = []
        positions  = []
        mask_list  = []

        for i in range(n_voxels):
            pt_mask = (inverse == i)
            voxel_pts = points[pt_mask]
            
            # PointNet-lite: 평균 집계
            feat = voxel_pts.mean(dim=0).to(self.device)
            emb  = self.point_encoder(feat.unsqueeze(0)).squeeze(0)
            embeddings.append(emb)
            
            # 위치: voxel 중심 좌표
            center = unique_voxels[i].float() * self.voxel_size
            positions.append(center)
            mask_list.append(True)

        matrix    = torch.stack(embeddings)     # [n_voxels × d]
        positions = torch.stack(positions)      # [n_voxels × 3]
        mask      = torch.tensor(mask_list, dtype=torch.bool)

        return ModalMatrix("lidar", matrix, positions, mask, 0.0)
```

---

### 2-3. Camera Encoder (V-JEPA tubelet)

```python
# htp/multimodal/encoders/camera_encoder.py

import torch
import torch.nn.functional as F
from ..modal_encoder import ModalEncoder, ModalMatrix


class CameraEncoder(ModalEncoder):
    """
    RGB 이미지/비디오 → ModalMatrix
    
    V-JEPA 방식:
      tubelet = 2프레임 × 16×16 픽셀 패치
      각 tubelet → embed_dim 벡터
    
    단일 프레임: [H × W × 3] → [(H/16) × (W/16) × embed_dim]
    비디오:      [T × H × W × 3] → [T/2 × (H/16) × (W/16) × embed_dim]
    
    3D-RoPE 단순화:
      각 패치의 (t, h, w) 위치를 sin/cos로 인코딩
    """

    def __init__(self,
                 embed_dim:   int = 64,
                 patch_size:  int = 16,   # 픽셀 단위
                 device:      str = "cpu"):
        super().__init__(embed_dim, device)
        self.patch_size = patch_size

        # 간단한 패치 인코더 (ViT patch embedding 단순화)
        self.patch_encoder = torch.nn.Linear(
            patch_size * patch_size * 3, embed_dim
        ).to(device)

    def encode(self, image: torch.Tensor) -> ModalMatrix:
        """
        image: [H × W × 3] or [T × H × W × 3]
        """
        import time

        # 단일 프레임으로 통일
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1 × H × W × 3]

        T, H, W, C = image.shape
        P = self.patch_size
        nH, nW = H // P, W // P

        embeddings = []
        positions  = []

        for t in range(T):
            for h in range(nH):
                for w in range(nW):
                    # 패치 추출
                    patch = image[t, h*P:(h+1)*P, w*P:(w+1)*P, :]
                    patch = patch.reshape(-1).float().to(self.device)

                    # 패치 인코딩
                    emb = self.patch_encoder(patch)
                    embeddings.append(emb)

                    # 3D-RoPE 단순화: sin/cos 위치 인코딩
                    pos = self._rope_encode(t, h, w)
                    positions.append(pos)

        matrix    = torch.stack(embeddings)   # [n_patches × d]
        positions = torch.stack(positions)    # [n_patches × 3]
        mask      = torch.ones(len(embeddings), dtype=torch.bool)

        return ModalMatrix("camera", matrix, positions, mask, time.time())

    def _rope_encode(self, t: int, h: int, w: int) -> torch.Tensor:
        """
        3D Rotary Position Encoding 단순화 버전
        
        V-JEPA 2의 3D-RoPE:
          각 tubelet의 (t, h, w) → 회전 행렬
        
        여기서는 sin/cos 인코딩으로 단순화:
          pos = [sin(t), cos(t), sin(h), cos(h), sin(w), cos(w)]
        """
        import math
        return torch.tensor([
            math.sin(t), math.cos(t),
            math.sin(h), math.cos(h),
            math.sin(w), math.cos(w),
        ], dtype=torch.float32)
```

---

### 2-4. Audio Encoder

```python
# htp/multimodal/encoders/audio_encoder.py

import torch
import math
from ..modal_encoder import ModalEncoder, ModalMatrix


class AudioEncoder(ModalEncoder):
    """
    오디오 → ModalMatrix
    
    입력: [samples] 또는 스펙트로그램 [freq × time]
    출력: [n_frames × embed_dim]
    
    V-JEPA 대응:
      오디오 프레임 = 시간 tubelet
      주파수 = 공간 차원 대응
    
    처리:
      1. STFT → 스펙트로그램 [freq × time]
      2. 시간 프레임 분할
      3. 각 프레임 → embed_dim
    """

    def __init__(self,
                 embed_dim:   int = 64,
                 n_fft:       int = 512,
                 hop_length:  int = 256,
                 frame_size:  int = 8,    # 프레임당 hop 수
                 device:      str = "cpu"):
        super().__init__(embed_dim, device)
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.frame_size = frame_size

        n_freq = n_fft // 2 + 1
        self.frame_encoder = torch.nn.Linear(
            n_freq * frame_size, embed_dim
        ).to(device)

    def encode(self, audio: torch.Tensor) -> ModalMatrix:
        """
        audio: [samples] 1D 오디오 또는 [freq × time] 스펙트로그램
        """
        import time

        if audio.dim() == 1:
            # STFT로 스펙트로그램 생성
            spec = torch.stft(
                audio.float(),
                n_fft      = self.n_fft,
                hop_length = self.hop_length,
                return_complex = True,
            )
            spec = spec.abs()   # [freq × time]
        else:
            spec = audio.float()  # 이미 스펙트로그램

        freq, time_steps = spec.shape
        n_frames = time_steps // self.frame_size

        embeddings = []
        positions  = []

        for i in range(n_frames):
            frame = spec[:, i*self.frame_size:(i+1)*self.frame_size]
            frame = frame.reshape(-1).to(self.device)

            if frame.shape[0] != self.frame_encoder.in_features:
                frame = F.pad(frame, (0, self.frame_encoder.in_features - frame.shape[0]))

            emb = self.frame_encoder(frame)
            embeddings.append(emb)

            # 위치: 시간 인덱스
            pos = torch.tensor([
                math.sin(i / n_frames * math.pi),
                math.cos(i / n_frames * math.pi),
                float(i) / max(n_frames, 1),
            ], dtype=torch.float32)
            positions.append(pos)

        if not embeddings:
            matrix    = torch.zeros(1, self.embed_dim, device=self.device)
            positions = torch.zeros(1, 3)
            mask      = torch.zeros(1, dtype=torch.bool)
        else:
            matrix    = torch.stack(embeddings)
            positions = torch.stack(positions)
            mask      = torch.ones(len(embeddings), dtype=torch.bool)

        return ModalMatrix("audio", matrix, positions, mask, time.time())
```

---

### 2-5. IMU / 센서 Encoder

```python
# htp/multimodal/encoders/imu_encoder.py

import torch
import math
from ..modal_encoder import ModalEncoder, ModalMatrix


class IMUEncoder(ModalEncoder):
    """
    IMU / 시계열 센서 → ModalMatrix
    
    입력: [channels × time] (가속도계, 자이로, 자력계 등)
    출력: [n_windows × embed_dim]
    
    처리:
      슬라이딩 윈도우로 시계열 분할
      각 윈도우 → embed_dim
    """

    def __init__(self,
                 embed_dim:    int = 64,
                 n_channels:   int = 6,     # ax,ay,az,gx,gy,gz
                 window_size:  int = 20,    # 윈도우 크기 (타임스텝)
                 stride:       int = 10,    # 슬라이딩 보폭
                 device:       str = "cpu"):
        super().__init__(embed_dim, device)
        self.window_size = window_size
        self.stride      = stride

        self.window_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_channels * window_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embed_dim),
        ).to(device)

    def encode(self, sensor_data: torch.Tensor) -> ModalMatrix:
        """
        sensor_data: [channels × time]
        """
        import time as time_module

        C, T = sensor_data.shape
        embeddings = []
        positions  = []

        for start in range(0, T - self.window_size + 1, self.stride):
            window = sensor_data[:, start:start + self.window_size]
            window = window.reshape(-1).float().to(self.device)

            emb = self.window_encoder(window)
            embeddings.append(emb)

            # 위치: 시간 중심
            t_center = (start + self.window_size / 2) / T
            pos = torch.tensor([
                math.sin(t_center * math.pi),
                math.cos(t_center * math.pi),
                t_center,
            ], dtype=torch.float32)
            positions.append(pos)

        if not embeddings:
            matrix    = torch.zeros(1, self.embed_dim, device=self.device)
            positions = torch.zeros(1, 3)
            mask      = torch.zeros(1, dtype=torch.bool)
        else:
            matrix    = torch.stack(embeddings)
            positions = torch.stack(positions)
            mask      = torch.ones(len(embeddings), dtype=torch.bool)

        return ModalMatrix("imu", matrix, positions, mask, time_module.time())
```

---

## 3. ModalRegionRuntime — V-JEPA Encoder Region

```python
# htp/multimodal/modal_region_runtime.py

from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from typing import Any

from ..runtime.region_runtime  import RegionRuntime
from ..runtime.htp_runtime     import RunResult
from ..thalamus.region_signal  import RegionSignal
from .modal_encoder            import ModalEncoder, ModalMatrix


class ModalRegionRuntime(RegionRuntime):
    """
    V-JEPA Encoder + Friston PredictiveRegion 결합
    
    역할:
      1. raw 센서 입력 → ModalMatrix (V-JEPA encoder)
      2. 예측 생성 → predict_next() (Friston)
      3. 예측 오차 계산 → precision 업데이트
      4. 오차 신호를 Thalamus로 전달
    
    각 모달리티가 이 클래스를 상속:
      LiDARRegion(ModalRegionRuntime, encoder=LiDAREncoder)
      CameraRegion(ModalRegionRuntime, encoder=CameraEncoder)
      AudioRegion(ModalRegionRuntime, encoder=AudioEncoder)
      IMURegion(ModalRegionRuntime, encoder=IMUEncoder)
    """

    def __init__(self,
                 region_name: str,
                 specialty:   str,
                 encoder:     ModalEncoder,
                 config       = None,
                 gen_config   = None):
        super().__init__(region_name, specialty, config, gen_config)
        self.encoder = encoder
        self.modality = specialty   # "lidar", "camera", "audio", etc.

        # 현재 모달 행렬 (Thalamus가 읽어감)
        self._current_modal_matrix: ModalMatrix | None = None

        # 예측 코딩 상태 (Friston)
        self._predicted_matrix: torch.Tensor | None = None
        self._last_error:       torch.Tensor | None = None

        # 모달리티별 집계 프로젝션 (행렬 → 벡터)
        self._pool_proj = torch.nn.Linear(
            encoder.embed_dim, encoder.embed_dim
        )

    # ── V-JEPA: 모달리티 인코딩 ────────────────────────

    def encode_modality(self, raw_input) -> ModalMatrix:
        """
        raw 입력 → ModalMatrix
        
        V-JEPA: 각 모달리티를 공통 잠재 공간으로 인코딩
        """
        modal_matrix = self.encoder.encode(raw_input)
        self._current_modal_matrix = modal_matrix
        return modal_matrix

    # ── Friston: 예측 생성 ──────────────────────────────

    def predict_next(self) -> torch.Tensor | None:
        """
        V-JEPA Predictor + Friston 예측 코딩
        
        현재 modal_matrix의 집계 벡터로 다음 상태 예측
        → Thalamus의 다음 스텝 통합에 활용
        """
        if self._current_modal_matrix is None:
            return None

        # 현재 행렬 집계 (mean pooling)
        mat     = self._current_modal_matrix.matrix
        pooled  = mat[self._current_modal_matrix.mask].mean(dim=0)
        predicted = self._pool_proj(pooled)
        self._predicted_matrix = predicted
        return predicted

    def compute_error(self, actual_pooled: torch.Tensor) -> torch.Tensor:
        """예측 오차 = actual - predicted"""
        if self._predicted_matrix is None:
            return torch.zeros_like(actual_pooled)
        n = min(len(actual_pooled), len(self._predicted_matrix))
        error = actual_pooled[:n] - self._predicted_matrix[:n]
        self.encoder.update_precision(error)
        self._last_error = error
        return error

    # ── 통합 run() ─────────────────────────────────────

    def run(self, data: Any, entry=None, max_depth: int = 8) -> RunResult:
        """
        data: raw 센서 입력 또는 {'modality': tensor} dict
        
        처리 순서:
          1. 모달리티 인코딩 (V-JEPA)
          2. 예측 생성
          3. HTP 노드 실행
          4. 예측 오차 계산 (Friston)
        """
        t0 = time.perf_counter()

        # 1. raw 입력 추출
        raw = self._extract_raw(data)

        # 2. V-JEPA 인코딩
        if raw is not None:
            modal_matrix = self.encode_modality(raw)
            # HTP 노드용 데이터로 변환 (집계 벡터)
            data_for_nodes = self._modal_to_dict(modal_matrix)
        else:
            data_for_nodes = data

        # 3. 예측 생성 (이전 상태 기반)
        self.predict_next()

        # 4. HTP 노드 실행
        result = super().run(data_for_nodes, entry, max_depth)

        # 5. 예측 오차 계산
        if self._current_modal_matrix is not None:
            mat    = self._current_modal_matrix.matrix
            valid  = self._current_modal_matrix.mask
            actual = mat[valid].mean(dim=0) if valid.any() else mat.mean(dim=0)
            self.compute_error(actual)

        result.total_ms = (time.perf_counter() - t0) * 1000
        return result

    def collect_signal(self) -> RegionSignal:
        """
        RegionSignal에 modal_matrix 포함
        Thalamus가 이 행렬로 cross-modal attention 수행
        """
        sig = super().collect_signal()

        # output_vec: 예측 오차 벡터 (Friston) or 모달 집계
        if self._last_error is not None:
            n = min(len(self._last_error), len(sig.output_vec))
            sig.output_vec[:n] = self._last_error[:n]

        # precision 주입 (Friston)
        sig.precision = self.encoder.precision

        # modal_matrix 첨부 (Thalamus Cross-modal Attention용)
        sig.modal_matrix = self._current_modal_matrix

        return sig

    # ── 내부 헬퍼 ─────────────────────────────────────

    def _extract_raw(self, data):
        """data에서 이 모달리티의 raw 입력 추출"""
        if isinstance(data, dict):
            return data.get(self.modality) or data.get("raw")
        if isinstance(data, torch.Tensor):
            return data
        return None

    def _modal_to_dict(self, modal_matrix: ModalMatrix) -> dict:
        """ModalMatrix → HTP 노드가 처리할 수 있는 dict"""
        mat    = modal_matrix.matrix
        valid  = modal_matrix.mask
        pooled = mat[valid].mean(dim=0) if valid.any() else mat.mean(dim=0)
        return {
            "modality": modal_matrix.modality,
            "embedding": pooled,
            "n_patches": int(valid.sum().item()),
            "label": self.modality,
        }
```

---

## 4. RegionSignal 확장 (modal_matrix 추가)

```python
# thalamus/region_signal.py 수정

@dataclass
class RegionSignal:
    """피질 Region → Thalamus 통신 단위 (멀티모달 확장)"""
    region_id:    str
    hub_strength: float
    fire_rate:    float
    top_hubs:     list
    overload:     bool
    output_vec:   torch.Tensor
    # Friston 추가 (Phase 2)
    precision:         float = 1.0
    prediction_error:  float = 0.0
    # V-JEPA 멀티모달 추가 ★
    modal_matrix:      "ModalMatrix | None" = None   # 모달 행렬
    modality:          str = "text"                   # 모달리티 타입
```

---

## 5. Thalamus — Cross-modal Fusion (Le MuMo JEPA)

```python
# thalamus/thalamus.py 확장

class MultimodalThalamus(Thalamus):
    """
    Le MuMo JEPA 방식 Cross-modal Fusion Thalamus
    
    기존 Thalamus: 각 Region의 output_vec만 처리
    확장:          각 Region의 modal_matrix로 cross-modal attention
    
    Le MuMo JEPA 핵심:
      1. 각 모달리티의 패치 행렬 수집
      2. 학습 가능한 Fusion Tokens로 cross-modal attention
      3. Pruning: 모달 토큰 제거 → fusion token만 남김
      4. 공유 잠재 벡터 생성 → state_vec
    
    Friston 결합:
      각 모달리티의 precision으로 attention 가중
    """

    def __init__(self, regions: dict,
                 temperature:    float = 1.0,
                 core_beta:      float = 5.0,
                 core_theta:     float = 0.3,
                 compress_dim:   int   = 64,
                 n_fusion_tokens:int   = 8,    # Le MuMo JEPA fusion tokens
                 embed_dim:      int   = 64):
        super().__init__(regions, temperature, core_beta, core_theta, compress_dim)
        self.n_fusion_tokens = n_fusion_tokens
        self.embed_dim       = embed_dim

        # Le MuMo JEPA: 학습 가능한 Fusion Tokens
        self.fusion_tokens = torch.nn.Parameter(
            torch.randn(n_fusion_tokens, embed_dim) * 0.02
        )

        # Cross-modal Attention (fusion tokens ← modal patches)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = 4,
            batch_first = True,
        )

        # 최종 압축: fusion tokens → state_vec
        self.fusion_proj = torch.nn.Linear(
            n_fusion_tokens * embed_dim, compress_dim
        )

    def step(self, data=None, top_down=None) -> "ThalamusOutput":
        """
        멀티모달 Thalamus step
        
        기존 JL Projection 대신:
          1. 각 Region의 modal_matrix 수집
          2. Cross-modal Attention (Fusion Tokens)
          3. Pruning → state_vec 생성
        """
        self._step += 1

        if not self.regions:
            raise RuntimeError("Region 없음")

        # 1. 각 Region 신호 수집
        signals = [r.collect_signal() for r in self.regions.values()]

        # 2. CoreCells: Precision 기반 게이팅 (Friston)
        gating = self.core.gate(signals, top_down=top_down)

        # 3. MatrixCells: WTA
        competition = self.matrix.compete(signals, gating)

        # 4. CoreCells Hebbian 학습
        self.core.update(
            winner_id = competition.winner_id,
            all_ids   = [s.region_id for s in signals],
        )

        # 5. NGE 트리거
        for sig in signals:
            if sig.overload:
                self.nge_trigger.fire(sig.region_id, sig.hub_strength)

        # 6. ★ Cross-modal Fusion (Le MuMo JEPA)
        state_vec = self._cross_modal_fusion(signals, gating)

        return ThalamusOutput(
            winner     = competition.winner_id,
            state_vec  = state_vec,
            gating     = gating,
            suppressed = competition.suppression_map,
            step       = self._step,
        )

    def _cross_modal_fusion(
        self,
        signals: list,
        gating:  "GatingMask",
    ) -> torch.Tensor:
        """
        Le MuMo JEPA Cross-modal Attention
        
        처리:
          1. 각 Region의 modal_matrix 수집
          2. Precision으로 가중 (Friston)
          3. Fusion Tokens가 모든 모달 패치에 attention
          4. Pruning: 모달 토큰 제거 → fusion token만 남김
          5. fusion tokens → state_vec (압축)
        
        수학:
          K = V = [modal_patches_1; modal_patches_2; ...]  -- 모든 모달 패치
          Q = fusion_tokens                                  -- 학습 가능
          fusion_out = Attention(Q, K, V)
          state_vec  = Linear(flatten(fusion_out))
        """
        # 모달 패치 수집 + precision 가중
        all_patches = []
        for sig in signals:
            gate_score = gating.scores.get(sig.region_id, 0.5)
            precision  = getattr(sig, 'precision', 1.0)
            weight     = gate_score * precision    # Friston 정밀도 가중

            if sig.modal_matrix is not None:
                mat   = sig.modal_matrix.matrix   # [n_patches × d]
                valid = sig.modal_matrix.mask
                if valid.any():
                    patches = mat[valid] * weight  # precision 가중
                    all_patches.append(patches)
            else:
                # modal_matrix 없는 Region: output_vec를 패치로 사용
                vec = sig.output_vec
                if vec.numel() >= self.embed_dim:
                    patch = vec[:self.embed_dim].unsqueeze(0) * weight
                else:
                    patch = F.pad(
                        vec.unsqueeze(0),
                        (0, self.embed_dim - vec.numel())
                    ) * weight
                all_patches.append(patch)

        if not all_patches:
            return torch.zeros(self.compress_dim)

        # 모든 모달 패치 통합: [total_patches × d]
        kv = torch.cat(all_patches, dim=0).unsqueeze(0)  # [1 × P × d]

        # Fusion Tokens: [1 × n_fusion × d]
        q = self.fusion_tokens.unsqueeze(0)

        # Cross-modal Attention (Q=fusion, K=V=모달 패치)
        fusion_out, _ = self.cross_attn(q, kv, kv)  # [1 × n_fusion × d]

        # Pruning: 모달 토큰 제거, fusion token만 남김 (Le MuMo JEPA)
        # → 이미 fusion_out만 남아있음 (pruning 완료)

        # 압축: fusion tokens → state_vec
        flat      = fusion_out.squeeze(0).reshape(-1)     # [n_fusion × d]
        state_vec = self.fusion_proj(flat)                 # [compress_dim]

        return state_vec.detach()
```

---

## 6. 사용 예시

```python
# 멀티모달 HTP 구성 예시

from htp.multimodal.encoders.lidar_encoder  import LiDAREncoder
from htp.multimodal.encoders.camera_encoder import CameraEncoder
from htp.multimodal.encoders.audio_encoder  import AudioEncoder
from htp.multimodal.encoders.imu_encoder    import IMUEncoder
from htp.multimodal.modal_region_runtime    import ModalRegionRuntime
from htp.thalamus.thalamus                  import MultimodalThalamus
from htp.runtime.brain_runtime              import BrainRuntime

# 1. 각 모달리티 Region 생성
lidar_region = ModalRegionRuntime(
    region_name = "lidar",
    specialty   = "lidar",
    encoder     = LiDAREncoder(embed_dim=64, voxel_size=0.5),
)

camera_region = ModalRegionRuntime(
    region_name = "camera",
    specialty   = "camera",
    encoder     = CameraEncoder(embed_dim=64, patch_size=16),
)

audio_region = ModalRegionRuntime(
    region_name = "audio",
    specialty   = "audio",
    encoder     = AudioEncoder(embed_dim=64),
)

imu_region = ModalRegionRuntime(
    region_name = "imu",
    specialty   = "imu",
    encoder     = IMUEncoder(embed_dim=64, n_channels=6),
)

# LLM 추론 Region (기존)
from htp.llm.llm_region_runtime import LLMRegionRuntime
reasoning_region = LLMRegionRuntime(
    region_name = "reasoning",
    specialty   = "reasoning",
    model       = "claude-sonnet-4-6",
    use_mock    = True,  # 테스트용
)

# 2. BrainRuntime 구성 (MultimodalThalamus 사용)
brain = BrainRuntime()
brain.add_region("lidar",     lidar_region)
brain.add_region("camera",    camera_region)
brain.add_region("audio",     audio_region)
brain.add_region("imu",       imu_region)
brain.add_region("reasoning", reasoning_region)

# MultimodalThalamus로 교체
brain.thalamus = MultimodalThalamus(
    regions         = brain.regions,
    compress_dim    = 64,
    n_fusion_tokens = 8,
    embed_dim       = 64,
)

# 3. 멀티모달 입력으로 실행
import torch
multimodal_input = {
    "lidar":  torch.randn(500, 4),              # 500 포인트
    "camera": torch.randn(224, 224, 3),         # RGB 이미지
    "audio":  torch.randn(16000),               # 1초 오디오
    "imu":    torch.randn(6, 100),              # 100 타임스텝
    "text":   "물체를 집어서 왼쪽에 놓아라",   # 텍스트 명령
}

action = brain.run(multimodal_input)
print(f"Action: {action.type} | Winner: {action.winner}")
print(f"Reason: {action.reason}")
```

---

## 7. 파일 구조

```
htp/
  multimodal/                    ★ 새로 추가
    __init__.py
    modal_encoder.py             # ModalEncoder 기반 클래스 + ModalMatrix
    modal_region_runtime.py      # V-JEPA + Friston 결합 Region
    encoders/
      __init__.py
      lidar_encoder.py           # LiDAR 포인트 클라우드
      camera_encoder.py          # RGB 이미지/비디오 (tubelet)
      audio_encoder.py           # 오디오 스펙트로그램
      imu_encoder.py             # IMU/시계열 센서
      text_encoder.py            # 텍스트 (기존 LLM 연계)
      tactile_encoder.py         # 촉각 센서 (Phase 4)

  thalamus/
    thalamus.py                  # MultimodalThalamus 추가
    region_signal.py             # modal_matrix 필드 추가

  runtime/
    brain_runtime.py             # MultimodalThalamus 지원
```

---

## 8. 수학 요약

```
모달리티 인코딩 (V-JEPA):
  raw_i → Encoder_i → M_i ∈ R^{n_i × d}
  (포인트/픽셀/프레임 → 패치 행렬)

3D-RoPE 위치 인코딩 (Camera):
  pos(t,h,w) = [sin(t), cos(t), sin(h), cos(h), sin(w), cos(w)]

Cross-modal Attention (Le MuMo JEPA):
  KV = concat([M_1; M_2; ...; M_k]) ∈ R^{P_total × d}
  Q  = FusionTokens ∈ R^{n_f × d}
  A  = softmax(QK^T / √d) V             -- Attention
  
Precision 가중 (Friston):
  KV_i = M_i × gate_i × π_i             -- 정밀도 가중

Pruning (Le MuMo JEPA):
  모달 토큰 제거 → FusionTokens만 남김
  
state_vec 생성:
  state_vec = Linear(flatten(A)) ∈ R^{64}

예측 오차 (Friston):
  ε_i = actual_pooled_i - predicted_i
  π_i = 0.9π_{i-1} + 0.1/Var(ε_i)
```

---

## 9. 로드맵

```
Phase 2 (즉시):
  ModalEncoder 기반 클래스
  LiDAREncoder, CameraEncoder
  RegionSignal.modal_matrix 추가
  MultimodalThalamus (Cross-modal Attention)

Phase 3:
  AudioEncoder, IMUEncoder
  ModalRegionRuntime.predict_next() (Friston 결합)
  3D-RoPE 완전 구현

Phase 4:
  TactileEncoder (촉각)
  V-JEPA 2 사전학습 가중치 로드
  Masked Prediction 학습 루프
  Action-conditioned 예측 (V-JEPA 2-AC 방식)
```

---

*V-JEPA 2 (Meta AI 2025) + Le MuMo JEPA (2025)*
*+ LeCun 검토 + Friston FEP 통합 설계*
