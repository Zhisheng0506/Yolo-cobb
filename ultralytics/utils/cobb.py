from __future__ import annotations

from dataclasses import dataclass

import torch

from ultralytics.utils.metrics import batch_probiou


@dataclass
class COBBConfig:
    ratio_type: str = "sig"
    pow_iou: float = 1.0
    eps: float = 1e-6


class COBBCoder:
    """PyTorch implementation of Continuous OBB (COBB)."""

    def __init__(self, ratio_type: str = "sig", pow_iou: float = 1.0, eps: float = 1e-6):
        self.cfg = COBBConfig(ratio_type=ratio_type, pow_iou=pow_iou, eps=eps)
        if self.cfg.ratio_type != "sig":
            raise NotImplementedError("Only ratio_type='sig' is currently supported.")

    def _clamp_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        return ratio.clamp(self.cfg.eps, 1.0 - self.cfg.eps)

    def encode_ratio(self, ratio_raw: torch.Tensor) -> torch.Tensor:
        ratio_raw = self._clamp_ratio(ratio_raw)
        if self.cfg.ratio_type == "sig":
            return 1.0 - torch.sqrt(torch.clamp(1.0 - ratio_raw, min=0.0))
        raise NotImplementedError

    def decode_ratio(self, ratio_pred: torch.Tensor) -> torch.Tensor:
        ratio_pred = ratio_pred.clamp(self.cfg.eps, 1.0 - self.cfg.eps)
        if self.cfg.ratio_type == "sig":
            return 1.0 - torch.square(1.0 - ratio_pred)
        raise NotImplementedError

    def rbox_to_poly(self, rbboxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h, theta = rbboxes.unbind(dim=-1)
        cos_t = torch.cos(theta) * 0.5
        sin_t = torch.sin(theta) * 0.5
        wx, wy = w * cos_t, w * sin_t
        hx, hy = -h * sin_t, h * cos_t
        p0 = torch.stack([cx + wx + hx, cy + wy + hy], dim=-1)
        p1 = torch.stack([cx + wx - hx, cy + wy - hy], dim=-1)
        p2 = torch.stack([cx - wx - hx, cy - wy - hy], dim=-1)
        p3 = torch.stack([cx - wx + hx, cy - wy + hy], dim=-1)
        return torch.stack([p0, p1, p2, p3], dim=-2)

    def poly_to_rbox(self, polys: torch.Tensor) -> torch.Tensor:
        center = polys.mean(dim=-2)
        edge1 = polys[:, 1] - polys[:, 0]
        edge2 = polys[:, 2] - polys[:, 1]
        width = torch.norm(edge1, dim=-1)
        height = torch.norm(edge2, dim=-1)
        angle = torch.atan2(edge1[:, 1], edge1[:, 0])
        return torch.stack([center[:, 0], center[:, 1], width, height, angle], dim=-1)

    @torch.no_grad()
    def encode(self, rbboxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rbboxes.numel() == 0:
            zeros = torch.zeros((0, 1), device=rbboxes.device, dtype=rbboxes.dtype)
            return zeros, torch.zeros((0, 4), device=rbboxes.device, dtype=rbboxes.dtype)
        polys = self.rbox_to_poly(rbboxes)
        hbboxes = torch.cat([polys.min(dim=-2).values, polys.max(dim=-2).values], dim=-1)
        ratio_raw, _ = self._ratio_from_poly(polys, hbboxes)
        ratio_target = self.encode_ratio(ratio_raw).unsqueeze(-1)
        candidates = self.build_candidates(hbboxes, ratio_raw)
        score_targets = torch.stack([batch_probiou(candidates[:, i], rbboxes) for i in range(4)], dim=-1).clamp_(0)
        score_targets = torch.pow(score_targets, self.cfg.pow_iou)
        return ratio_target, score_targets

    def _ratio_from_poly(self, polys: torch.Tensor, hbboxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_min, y_min, x_max, y_max = hbboxes.unbind(dim=-1)
        w = (x_max - x_min).clamp_min(self.cfg.eps)
        h = (y_max - y_min).clamp_min(self.cfg.eps)
        x_order = polys[..., 0].argsort(dim=1)
        y_order = polys[..., 1].argsort(dim=1)
        idx = torch.arange(polys.shape[0], device=polys.device)
        s_x = polys[idx, x_order[:, 1], 0]
        s_y = polys[idx, y_order[:, 1], 1]
        dx = (s_x - x_min) / w
        dy = (s_y - y_min) / h
        w_large = w > h
        ratio = torch.zeros_like(dx)
        ratio[~w_large] = dx[~w_large] * (1 - dx[~w_large]) * 4
        ratio[w_large] = dy[w_large] * (1 - dy[w_large]) * 4
        return ratio, w_large

    def build_candidates(self, hbboxes: torch.Tensor, ratio_raw: torch.Tensor) -> torch.Tensor:
        x_min, y_min, x_max, y_max = hbboxes.unbind(dim=-1)
        w = (x_max - x_min).clamp_min(self.cfg.eps)
        h = (y_max - y_min).clamp_min(self.cfg.eps)
        ratio = self._clamp_ratio(ratio_raw)
        base = ratio / 4.0

        w_large = w > h
        x1 = torch.zeros_like(ratio)
        x2 = torch.zeros_like(ratio)
        y1 = torch.zeros_like(ratio)
        y2 = torch.zeros_like(ratio)

        idx = ~w_large
        delta = torch.sqrt(torch.clamp(1.0 - 4.0 * base[idx], min=0.0))
        x1[idx] = (1.0 - delta) * 0.5 * w[idx]
        x2[idx] = (1.0 + delta) * 0.5 * w[idx]
        coeff = torch.square(w[idx] / h[idx])
        delta_y = torch.sqrt(torch.clamp(1.0 - 4.0 * coeff * base[idx], min=0.0))
        y1[idx] = (1.0 - delta_y) * 0.5 * h[idx]
        y2[idx] = (1.0 + delta_y) * 0.5 * h[idx]

        idx = w_large
        delta = torch.sqrt(torch.clamp(1.0 - 4.0 * base[idx], min=0.0))
        y1[idx] = (1.0 - delta) * 0.5 * h[idx]
        y2[idx] = (1.0 + delta) * 0.5 * h[idx]
        coeff = torch.square(h[idx] / w[idx])
        delta_x = torch.sqrt(torch.clamp(1.0 - 4.0 * coeff * base[idx], min=0.0))
        x1[idx] = (1.0 - delta_x) * 0.5 * w[idx]
        x2[idx] = (1.0 + delta_x) * 0.5 * w[idx]

        poly1 = torch.stack(
            [
                torch.stack([x_min + x1, y_min], -1),
                torch.stack([x_max, y_min + y2], -1),
                torch.stack([x_max - x1, y_max], -1),
                torch.stack([x_min, y_max - y2], -1),
            ],
            dim=1,
        )
        poly2 = torch.stack(
            [
                torch.stack([x_min + x2, y_min], -1),
                torch.stack([x_max, y_min + y2], -1),
                torch.stack([x_max - x2, y_max], -1),
                torch.stack([x_min, y_max - y2], -1),
            ],
            dim=1,
        )
        poly3 = torch.stack(
            [
                torch.stack([x_min + x1, y_min], -1),
                torch.stack([x_max, y_min + y1], -1),
                torch.stack([x_max - x1, y_max], -1),
                torch.stack([x_min, y_max - y1], -1),
            ],
            dim=1,
        )
        poly4 = torch.stack(
            [
                torch.stack([x_min + x2, y_min], -1),
                torch.stack([x_max, y_min + y1], -1),
                torch.stack([x_max - x2, y_max], -1),
                torch.stack([x_min, y_max - y1], -1),
            ],
            dim=1,
        )

        cand = torch.stack(
            [
                self.poly_to_rbox(poly1.view(-1, 4, 2)),
                self.poly_to_rbox(poly2.view(-1, 4, 2)),
                self.poly_to_rbox(poly3.view(-1, 4, 2)),
                self.poly_to_rbox(poly4.view(-1, 4, 2)),
            ],
            dim=1,
        )
        return cand

    def decode(self, hbboxes: torch.Tensor, ratio_pred: torch.Tensor, score_pred: torch.Tensor) -> torch.Tensor:
        ratio_raw = self.decode_ratio(ratio_pred.squeeze(-1))
        candidates = self.build_candidates(hbboxes, ratio_raw)
        probs = torch.softmax(score_pred, dim=-1)
        best = torch.argmax(probs, dim=-1)
        idx = torch.arange(best.shape[0], device=best.device)
        return candidates[idx, best]

    def mix_candidates(self, hbboxes: torch.Tensor, ratio_pred: torch.Tensor, score_pred: torch.Tensor) -> torch.Tensor:
        ratio_raw = self.decode_ratio(ratio_pred.squeeze(-1))
        candidates = self.build_candidates(hbboxes, ratio_raw)
        weights = torch.softmax(score_pred, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.cfg.eps)
        rbboxes = torch.sum(weights.unsqueeze(-1) * candidates, dim=1)
        return rbboxes

