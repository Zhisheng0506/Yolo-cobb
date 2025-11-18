# COBB 方法解读与集成要点

## 1. 理论动机与连续性目标

- 传统 OBB 使用 `(x_c, y_c, w, h, \theta)`；当目标接近正方形或旋转角接近±90° 时，`w/h` 与角度会发生离散跳变，回归损失不连续，甚至出现 Decoding Incompleteness (DI) 与 Decoding Ambiguity (DA)，导致同一物体存在多种数值表示（README 中“完全解决 OBB 表示连续性”即针对该问题）。
- COBB（Continuous OBB）用“水平外接框 + 单一比例因子 + 类型评分”来替代角度：无论目标如何旋转，其水平包围盒是唯一的，只要再记录外接框内部对角线交点在短边方向上的相对位置，就可以恢复 OBB，避免角度周期性与宽高交换。
- 该比例定义为 `r = 4·s·(1-s)`，其中 `s∈[0,1]` 表示交点在短边上的归一化坐标；函数在 (0,1) 内连续、单峰，对称性使得长宽互换时仍对应同一 `r`，从而保持回归连续。
- 由于一个水平外接框和一个比例可以恢复四个镜像 OBB（关于长/短边对称），COBB 额外预测一个 4 维 score，训练时用真实 OBB 与四个候选的 IoU 作为软标签，推理时根据得分选择唯一实例，彻底消除 DA。

## 2. `COBBCoder` 编码 / 解码流程

### 2.1 编码：从旋转框到 `(ratio, score)`

- 输入 `rbboxes (N×5)`，先转为八点多边形再求包围盒，计算宽高并按长短边分两类。获取多边形在短边方向上的第二小坐标（`s`），然后写成 `ratio = 4·s·(1-s)`。当采用 `'sig'` 类型时再通过 `1-√(1-ratio)` 压缩动态范围，`'ln'` 类型则进行对数拉伸并对“方形”类型单独翻转，提升小目标/近方形的解码稳定性：

```149:199:python/jdet/models/boxes/cobb_coder.py
        ratio[jt.logical_not(w_large)] = h_large_dx * (1 - h_large_dx) * 4
        ratio[w_large] = w_large_dy * (1 - w_large_dy) * 4
        ...
        if self.ratio_type == 'sig':
            ratio = 1 - jt.sqrt(1 - ratio)
        elif self.ratio_type == 'ln':
            ...
            ratio = 1 + jt.log2(ratio)
```

- `rtype`（0~3）通过比较顶点的 x/y 排序位置确定是哪一种镜像关系；随后利用 `build_iou_matrix` 计算四种候选与真实框的 IoU，作为 `score_targets`。得分在训练时由 `SmoothL1Loss` 回归，使网络不仅学习类别，还学习“正确候选编号”。

### 2.2 解码：根据 `(hbboxes, ratio_pred, score_pred)` 恢复唯一 OBB

- 推理时先把预测的 ratio 逆映射回 `s`，再解二次方程得到横纵两个交点，从而生成四个候选多边形并转回旋转框（`build_polypairs`）：

```88:147:python/jdet/models/boxes/cobb_coder.py
        h_large_delta_x = jt.sqrt(1 - 4 * h_large_ratio)
        x1[h_large] = (1 - h_large_delta_x) / 2 * h_large_w
        ...
        poly1 = jt.stack([min_x+x1, min_y, max_x, min_y+y2, ...], dim=-1)
        ...
        return [poly_to_rotated_box(poly1), ..., poly4]
```

- 将四个候选拼成 `num_hbbox×4` 的 proposals，使用 `score_pred`（维度 4）挑出得分最高的类型即可恢复唯一旋转框：

```201:226:python/jdet/models/boxes/cobb_coder.py
        rbboxes_list = self.build_polypairs(hbboxes, bboxes_pred)
        rbboxes_proposals = jt.concat([rbboxes[:, None, :] for rbboxes in rbboxes_list], dim=1)
        best_index = jt.argmax(rotated_scores, dim=-1, keepdims=False)[0] + ...
        best_rbboxes = rbboxes_proposals[best_index, :]
```

- `RotatedCOBBCoder` 还提供 anchor 轴 ↔ 全局轴的旋转变换，以兼容带角度锚框的方法（如 RoI Transformer），其核心是对中心点做旋转平移，再复用 `COBBCoder`。

## 3. `COBB RoI Head` 的结构与训练

### 3.1 结构扩展

- 该 RoI Head 继承 `ConvFCRoIHead`，在共享骨干后额外添加 score/ratio 两套 Conv+FC 分支，并各自接一个 `Linear` 输出，尺寸分别为 4（score_dim）和 1（ratio_dim）。在默认设置下这两个分支复用共享特征，不与分类/回归层共享权重：

```51:85:python/jdet/models/roi_heads/cobb_roi_head.py
        self.score_convs, self.score_fcs, self.score_last_dim = ...
        self.fc_score = nn.Linear(self.score_last_dim, out_dim_score)
        ...
        self.fc_ratio = nn.Linear(self.score_last_dim, out_dim_ratio)
```

- `score_type`/`ratio_type` 决定激活函数（cos、sigmoid、softmax 等）。默认 `score_type='sigmoid'`、`ratio_type='sigmoid'` 以匹配 `COBBCoder` 的数值域。

### 3.2 监督信号

- `get_targets_single` 在采样到的正样本上调用 `self.cobb_coder.encode(rboxes)`，得到 `ratio_targets (N×1)` 和 `score_targets (N×4)`；对应的损失由 `loss_ratio`、`loss_score` 控制，默认 `SmoothL1`，并设置较大的权重（ratio 16.0）以强化几何约束：

```203:207:python/jdet/models/roi_heads/cobb_roi_head.py
            pos_ratio_targets, pos_score_targets = self.cobb_coder.encode(rboxes)
            ratio_targets[:num_pos, :] = pos_ratio_targets
            score_targets[:num_pos, :] = pos_score_targets
```

- 在 `loss` 中，仅对正样本的三套输出（bbox、score、ratio）计算损失，确保网络学习“水平框回归 + COBB 参数”联合表示。

### 3.3 推理与提案精炼

- `get_det_bboxes_single` 首先用常规 bbox coder 将 RoI 变为水平框，再交给 `cobb_coder.decode` 生成旋转框，随后走 `multiclass_nms_rotated`。因此只需要一处修改即可在所有阶段获得 OBB：

```266:295:python/jdet/models/roi_heads/cobb_roi_head.py
        bboxes = self.bbox_coder.decode(proposals, bbox_pred, target['img_size'][::-1])
        rbboxes = self.cobb_coder.decode(bboxes, ratio_pred, score_pred)
```

- `get_refine_proposals` 同样调用 `decode`，可在两阶段检测器内部迭代 refine。对于 anchor-based 版本，`SharedCOBBRoIHeadRbboxes` 则直接使用 `RotatedCOBBCoder`，避免水平框中间态。

## 4. YOLO 系列集成提示

1. **预测头输出**：在现有 `bbox_reg`（通常为 `(dx, dy, dw, dh, dθ)` 或四点坐标）之外，再增加一个标量 `ratio` 和一个 4 维 `score`。`ratio` 的激活需与 `COBBCoder` 的 `ratio_type` 对应（如 `sigmoid`）。`score` 可用 `sigmoid` 或 `softmax` 但需在 loss 中和 IoU 目标对齐。
2. **标签生成**：对于每个正样本，基于 GT 旋转框调用 `COBBCoder.encode` 得到 `ratio_target` 与 `score_target`。如果 YOLO 仍在水平框空间进行回归，可直接复用 `COBBCoder.encode`；若是 anchor-free 且直接输出中心点 + 尺寸，则需将预测框转换成 `rbbox` 以供编码。
3. **损失函数**：保持与 JDet 相同的 SmoothL1 或者改为 BCE / KL，但要注意 `score_target` 是 IoU 分布（非 one-hot），因此用 L1/L2 更合适。`ratio` 建议乘以较大 loss weight，以避免被分类/置信度掩盖。
4. **解码 / NMS**：推理阶段先得到水平框，再用 `COBBCoder.decode(hbbox, ratio_pred, score_pred)` 生成旋转框，之后才能进入旋转 NMS（如 mmrotate / JDet 的 `multiclass_nms_rotated`）。若 YOLO 使用多尺度 head，需要在每个 head 上统一调用 decode。
5. **Rotated Anchor 兼容**：如果 YOLO 版本（如 YOLOv5-OBB）已经预测角度，可将其角度分支替换成 `ratio+score`，或者借助 `RotatedCOBBCoder` 把锚框映射到局部坐标再编码，保证与现有训练逻辑最小改动。
6. **推理速度与部署**：`build_polypairs` 仅依赖基础算子（加减乘除、sqrt），可在 TensorRT/ONNX 中实现；需要注意 `jt.sqrt(1 - 4r)` 对 r>0.25 时数值敏感，训练时务必 clamp 到 (0,0.25]，否则会出现 NaN，这一点在 `decode` 中通过 `jt.clamp` 已处理，迁移时也要保持。

通过以上步骤，COBB 的“水平框 + 比例 + 类型”表示即可无缝嫁接到 YOLO，使其在任意朝向、极端长宽比目标上保持连续稳定的回归表现。\*\*\*
