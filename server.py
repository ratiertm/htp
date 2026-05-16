"""
Hub Topology Runtime — WebSocket Server
FastAPI로 HTPRuntime을 돌리고 Three.js 대시보드에 상태를 브로드캐스트한다.
"""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from typing import Set

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

from htp.runtime.htp_runtime import (
    HTPConfig, WeightMatrix, HubFormationEngine, PruningEngine,
)

# ─────────────────────────────────────────────────────────
# 엔진 초기화
# ─────────────────────────────────────────────────────────

cfg = HTPConfig(
    n_nodes         = 64,
    threshold       = 0.45,
    hebbian_lr      = 0.08,
    decay_rate      = 0.005,
    prune_threshold = 0.02,
)


def _build_engine(config: HTPConfig):
    wm  = WeightMatrix(config.n_nodes, config.device)
    hfe = HubFormationEngine(wm, config)
    pe  = PruningEngine(wm, hfe, config)
    return wm, hfe, pe


wm, hfe, pe = _build_engine(cfg)

sim = {
    "running": True,
    "speed":   0.12,
    "pattern": 0,
    "pruned":  0,     # 누적 가지치기 수
}

clients: Set[WebSocket] = set()


# ─────────────────────────────────────────────────────────
# 4D 피보나치 구 포지션
# ─────────────────────────────────────────────────────────

def fibonacci_4sphere(n: int, r3: float = 5.5):
    golden = math.pi * (3.0 - math.sqrt(5.0))
    silver = math.pi * (3.0 - math.sqrt(2.0))

    pos3d, pos4d = [], []
    for i in range(n):
        y    = 1.0 - (i / float(n - 1)) * 2.0
        rxy  = math.sqrt(max(0.0, 1.0 - y * y))
        th   = golden * i
        x    = math.cos(th) * rxy
        z    = math.sin(th) * rxy
        w    = math.sin(silver * i)
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        xn, yn, zn, wn = x/norm, y/norm, z/norm, w/norm

        pos3d.append([round(xn * r3, 3), round(yn * r3, 3), round(zn * r3, 3)])
        pos4d.append([round(xn, 5), round(yn, 5), round(zn, 5), round(wn, 5)])

    return pos3d, pos4d


NODE_POSITIONS, NODE_POS4D = fibonacci_4sphere(cfg.n_nodes)


# ─────────────────────────────────────────────────────────
# 신호 생성
# ─────────────────────────────────────────────────────────

def make_signal(step: int, pattern: int) -> torch.Tensor:
    dev  = wm.dev
    n    = cfg.n_nodes
    sig  = torch.zeros(n, device=dev)
    mode = pattern if pattern != 0 else (step % 9 // 3) + 1

    if mode == 1:
        sig[:8] = torch.rand(8, device=dev) * 0.8 + 0.4
    elif mode == 2:
        sig[:4]    = torch.rand(4, device=dev) * 0.7 + 0.3
        sig[16:20] = torch.rand(4, device=dev) * 0.7 + 0.3
    else:
        sig = torch.rand(n, device=dev) * 0.4
    return sig


# ─────────────────────────────────────────────────────────
# 엔진 스텝
# ─────────────────────────────────────────────────────────

def engine_step() -> str:
    signal = make_signal(hfe.step_count, sim["pattern"])
    fired  = hfe.step(signal)

    # 가지치기
    pruned_map  = pe.run_all(hfe.step_count)
    step_pruned = sum(pruned_map.values())
    sim["pruned"] += step_pruned

    W         = wm.W
    nz        = (W > 0.001).nonzero(as_tuple=False)
    n_edges   = len(nz)
    avg_w     = float(W[W > 0.001].mean()) if n_edges else 0.0

    hub_ids   = hfe.is_hub.nonzero(as_tuple=True)[0].tolist()
    fire_cnt  = [round(float(v), 1) for v in hfe.fire_count.tolist()]
    in_str    = [round(float(v), 3) for v in W.sum(dim=0).tolist()]
    out_str   = [round(float(v), 3) for v in W.sum(dim=1).tolist()]

    flat_edges = []
    for idx in nz:
        i, j = int(idx[0]), int(idx[1])
        flat_edges.extend([i, j, round(float(W[i, j]), 3)])

    return json.dumps({
        "type":       "step",
        "step":       hfe.step_count,
        "fired":      [round(float(v), 3) for v in fired.tolist()],
        "hubs":       hub_ids,
        "n_hubs":     len(hub_ids),
        "n_edges":    n_edges,
        "pruned":     sim["pruned"],
        "avg_weight": round(avg_w, 4),
        "edges":      flat_edges,
        "fire_count": fire_cnt,
        "in_str":     in_str,
        "out_str":    out_str,
    })


def reset_engine():
    global wm, hfe, pe
    wm, hfe, pe = _build_engine(cfg)
    sim["pruned"] = 0


# ─────────────────────────────────────────────────────────
# 브로드캐스트
# ─────────────────────────────────────────────────────────

async def broadcast(msg: str) -> None:
    dead: Set[WebSocket] = set()
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ─────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_engine_loop())
    yield

app = FastAPI(lifespan=lifespan)


async def _engine_loop():
    while True:
        if sim["running"] and clients:
            await broadcast(engine_step())
        await asyncio.sleep(sim["speed"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    # 초기 데이터 전송
    await websocket.send_text(json.dumps({
        "type":      "init",
        "n_nodes":   cfg.n_nodes,
        "positions": NODE_POSITIONS,
        "pos4d":     NODE_POS4D,
        "config": {
            "threshold":        cfg.threshold,
            "hub_pr_threshold": cfg.hub_pr_threshold,
            "hebbian_lr":       cfg.hebbian_lr,
            "decay_rate":       cfg.decay_rate,
            "n_nodes":          cfg.n_nodes,
        },
    }))

    try:
        async for text in websocket.iter_text():
            cmd    = json.loads(text)
            action = cmd.get("action")

            if action == "pause":
                sim["running"] = False
            elif action == "resume":
                sim["running"] = True
            elif action == "step":
                await broadcast(engine_step())
            elif action == "speed":
                sim["speed"] = max(0.02, min(2.0, float(cmd.get("value", 0.12))))
            elif action == "pattern":
                sim["pattern"] = int(cmd.get("value", 0))
            elif action == "reset":
                reset_engine()
                await broadcast(json.dumps({"type": "reset"}))

    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)
