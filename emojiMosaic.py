#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

按照“输入图块覆盖”的方式拼接 emoji 马赛克：
- 先对输入图做两次 3×3 卷积（盒式/高斯，可选 same/valid），
- 以 block×block 的输入区域为单位采样颜色（alpha 加权平均），
- 用 rgb888 字典里最近的颜色对应的 emoji 贴到输出图，
- 贴图的位置按输入坐标 * scale 缩放；scale = emoji_size / block，
  因此当 step < block 时，输出图上 tile 会产生重叠（按需求“不用担心重叠”）。

字典文件每行： RRGGBB  相对文件名
例如： E2DDE8  0043_cloud_32.png

示例：
  python emoji_mosaic_block_cover.py \
    --dict emoji_colors_rgb888.txt \
    --emoji-dir out_emoji \
    --input input.png \
    --out mosaic.png \
    --kernel box --passes 2 --conv-mode same \
    --block 16 --step 16 --emoji-size 32 --anchor center

  # 允许重叠（step < block），tile 会更密集：
  python emoji_mosaic_block_cover.py \
    --dict emoji_colors_rgb888.txt \
    --emoji-dir out_emoji \
    --input input.png \
    --out mosaic_overlap.png \
    --block 16 --step 8 --emoji-size 32
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy.spatial import cKDTree

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============== 工具 & 加速 ==============

def load_palette(dict_path: str) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """读取 rgb888 字典 -> (K,3) uint8 颜色数组 + [(HEX, filename)]"""
    colors, meta = [], []
    with open(dict_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"): continue
            parts = s.split()
            if len(parts) < 2: continue
            hexv = parts[0].upper()
            name = " ".join(parts[1:])
            if len(hexv) != 6: continue
            try:
                r = int(hexv[0:2], 16);
                g = int(hexv[2:4], 16);
                b = int(hexv[4:6], 16)
            except ValueError:
                continue
            colors.append([r, g, b]);
            meta.append((hexv, name))
    if not colors:
        print("⚠ 字典为空或无有效颜色。", file=sys.stderr);
        sys.exit(1)
    return np.array(colors, dtype=np.uint8), meta


def _kernel_3x3(kind: str) -> ImageFilter.Kernel:
    if kind == "box":
        w, scale = [1, 1, 1, 1, 1, 1, 1, 1, 1], 9
    elif kind == "gauss":
        w, scale = [1, 2, 1, 2, 4, 2, 1, 2, 1], 16
    else:
        raise ValueError("kernel must be 'box' or 'gauss'")
    return ImageFilter.Kernel((3, 3), w, scale=scale, offset=0)


def convolve_rgba(img_rgba: Image.Image, kernel: str, passes: int, mode: str) -> Image.Image:
    """仅对 RGB 做卷积，A 通道保持；mode='valid' 会每次四边裁 1px。"""
    if passes <= 0: return img_rgba
    K = _kernel_3x3(kernel)
    r, g, b, a = img_rgba.split()
    for _ in range(passes):
        r = r.filter(K);
        g = g.filter(K);
        b = b.filter(K)
    out = Image.merge("RGBA", (r, g, b, a))
    if mode == "valid":
        crop = passes
        w, h = out.size
        if w <= 2 * crop or h <= 2 * crop:
            raise ValueError("图太小，无法进行 valid 裁剪")
        out = out.crop((crop, crop, w - crop, h - crop))
    elif mode != "same":
        raise ValueError("conv-mode 只能 same/valid")
    return out


def build_tile_cache(palette_rgb: np.ndarray, meta: List[Tuple[str, str]],
                     emoji_dir: str, emoji_size: int):
    """为每个调色板索引准备一个 tile（RGBA, emoji_size×emoji_size）。找不到则用纯色方块兜底。"""
    cache = []
    for idx, (hexv, fname) in enumerate(meta):
        path = os.path.join(emoji_dir, fname)
        if os.path.exists(path):
            try:
                im = Image.open(path).convert("RGBA")
                im = im.resize((emoji_size, emoji_size), resample=Image.LANCZOS)
            except Exception:
                im = None
        else:
            im = None
        if im is None:
            r, g, b = palette_rgb[idx].tolist()
            im = Image.new("RGBA", (emoji_size, emoji_size), (r, g, b, 255))
        cache.append(im)
    return cache  # list[Image]


class NearestSearcher:
    """
    在 palette_rgb 中做最近邻搜索。
    新增参数 random_factor 用来引入随机性：

    - random_factor = 0.0: 完全最近邻（兼容原逻辑）
    - 0 < random_factor <= 1.0:
        对每个点，以 (1 - random_factor) 概率使用最近的颜色，
        以 random_factor 概率在若干个最近邻中随机选一个稍远的。
    """

    def __init__(self, palette_rgb: np.ndarray):
        self.palette = palette_rgb.astype(np.float32)  # (K,3)
        self.K = self.palette.shape[0]
        if self.K <= 0:
            raise ValueError("Palette is empty")

        if _HAVE_SCIPY:
            self.tree = cKDTree(self.palette)
        else:
            self.tree = None
        # 预计算 ||c||^2，给 fallback 用
        self.pal_sq = (self.palette * self.palette).sum(axis=1).astype(np.float32)

    def query(
            self,
            pixels_rgb: np.ndarray,
            mem_mb: float = 512.0,
            random_factor: float = 0.0,
            k_random: int = 3,
    ) -> np.ndarray:
        """
        pixels_rgb: (N,3) uint8 / float
        random_factor: [0,1] 之间的随机强度
            - 0: 完全最近邻
            - 0.1: 大约 10% 的像素会换成 2~k_random 之间的邻居
        k_random: 最多考虑多少个最近邻做随机（含最近的那个）

        返回: (N,) int 索引
        """
        if pixels_rgb.size == 0:
            return np.empty((0,), dtype=np.int32)

        X = pixels_rgb.astype(np.float32)
        N = X.shape[0]
        rng = np.random.default_rng()

        # 限制一下 k_random 的范围
        k = int(max(1, min(k_random, self.K)))
        random_factor = float(max(0.0, min(1.0, random_factor)))

        # ---------- 有 SciPy / KDTree 的情况 ----------
        if self.tree is not None:
            if random_factor <= 0.0 or k == 1:
                # 纯最近邻，和原逻辑完全一样
                _, idx = self.tree.query(X, k=1, workers=-1)
                return idx.astype(np.int32)

            # 要随机，就查前 k 个最近邻
            dists, idxs = self.tree.query(X, k=k, workers=-1)  # (N,k)
            if k == 1:
                return idxs.astype(np.int32)

            # 对每个样本，决定用第几个候选
            # 默认用第 0 个（最近）
            choice = np.zeros(N, dtype=np.int64)
            if random_factor > 0.0 and k > 1:
                mask = rng.random(N) < random_factor  # True 的那些要“乱来一下”
                num_alt = int(mask.sum())
                if num_alt > 0:
                    # 在 [1, k-1] 之间随便挑一个
                    alt_choice = rng.integers(1, k, size=num_alt)
                    choice[mask] = alt_choice

            final_idx = idxs[np.arange(N), choice]
            return final_idx.astype(np.int32)

        # ---------- 无 SciPy，NumPy fallback ----------
        # 和原来一样，按批处理，避免一次性爆内存
        # 估算一下每批大小
        bytes_per_float = 4  # float32
        # 每个批次要存 d2: (batch_size, K)，大概 batch_size*K*4 字节
        approx_bytes_per_item = self.K * bytes_per_float
        B = int((mem_mb * (1024 ** 2)) / max(approx_bytes_per_item, 1))
        B = max(1024, min(B, 65536))  # 人为夹一夹范围

        result = np.empty((N,), dtype=np.int32)
        X_sq = (X * X).sum(axis=1).astype(np.float32)  # (N,)

        for start in range(0, N, B):
            end = min(start + B, N)
            chunk = X[start:end]  # (b,3)
            chunk_sq = X_sq[start:end]  # (b,)

            # d2[i,j] = ||x_i||^2 - 2 x_i·c_j + ||c_j||^2
            dot = chunk @ self.palette.T  # (b,K)
            d2 = (chunk_sq[:, None] - 2.0 * dot + self.pal_sq[None, :])

            bsize = end - start

            if random_factor <= 0.0 or k == 1:
                # 纯最近邻（原逻辑）
                best_idx = np.argmin(d2, axis=1)
                result[start:end] = best_idx.astype(np.int32)
                continue

            # 需要随机：找出每行前 k 个最小值的索引
            k_eff = min(k, self.K)
            # argpartition 取出 k_eff 个最小值的“无序”索引
            topk_idx = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]  # (b,k_eff)

            # 按真实距离再排个序，让第 0 个就是最近的
            topk_d2 = np.take_along_axis(d2, topk_idx, axis=1)  # (b,k_eff)
            order = np.argsort(topk_d2, axis=1)  # (b,k_eff)
            topk_sorted = np.take_along_axis(topk_idx, order, axis=1)  # (b,k_eff)

            # 默认都用最近的
            chosen = topk_sorted[:, 0].copy()

            if random_factor > 0.0 and k_eff > 1:
                m = rng.random(bsize) < random_factor
                num_alt = int(m.sum())
                if num_alt > 0:
                    # 从 1...(k_eff-1) 之间随机挑
                    alt_choice = rng.integers(1, k_eff, size=num_alt)
                    chosen[m] = topk_sorted[m, alt_choice]

            result[start:end] = chosen.astype(np.int32)

        return result


# ====== 快速“块平均色”（alpha 加权）======

def _integral(img: np.ndarray) -> np.ndarray:
    """二维积分图：H×W -> 同尺寸前缀和（float64避免溢出）"""
    return img.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(I: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """用积分图求 [y0:y1, x0:x1) 的和"""
    res = I[y1 - 1, x1 - 1]
    if x0 > 0: res -= I[y1 - 1, x0 - 1]
    if y0 > 0: res -= I[y0 - 1, x1 - 1]
    if x0 > 0 and y0 > 0: res += I[y0 - 1, x0 - 1]
    return float(res)


def block_mean_colors_alpha(rgb: np.ndarray, alpha: np.ndarray,
                            blocks: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    对给定 block 列表计算 alpha 加权平均颜色。
    rgb: H×W×3 uint8, alpha: H×W uint8
    blocks: [(x0,y0,x1,y1), ...] 以输入坐标给出（半开区间）
    返回 (M,3) float32 颜色
    """
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)
    A = alpha.astype(np.float64)

    RA, GA, BA = R * A, G * A, B * A
    I_RA, I_GA, I_BA, I_A = map(_integral, (RA, GA, BA, A))

    out = np.zeros((len(blocks), 3), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(blocks):
        sumA = _rect_sum(I_A, x0, y0, x1, y1)
        if sumA <= 1e-6:
            out[i] = (255.0, 255.0, 255.0)  # 全透明块兜底白色
        else:
            r = _rect_sum(I_RA, x0, y0, x1, y1) / sumA
            g = _rect_sum(I_GA, x0, y0, x1, y1) / sumA
            b = _rect_sum(I_BA, x0, y0, x1, y1) / sumA
            out[i] = (r, g, b)
    return out.clip(0, 255)


# ============== 主流程（块覆盖 + 可重叠） ==============

def process(
        dict_path: str, emoji_dir: str, input_path: str, out_path: str,
        kernel: str, passes: int, conv_mode: str,
        block: int, step: Optional[int], emoji_size: int,
        anchor: str, alpha_threshold: int,
        random_factor: float,
        random_k: int,

):
    # 1) 字典 & 贴图缓存 & 最近邻
    palette, meta = load_palette(dict_path)  # (K,3)
    searcher = NearestSearcher(palette)
    tile_cache = build_tile_cache(palette, meta, emoji_dir, emoji_size)

    # 2) 读图 + 卷积
    img = Image.open(input_path).convert("RGBA")
    img_f = convolve_rgba(img, kernel=kernel, passes=passes, mode=conv_mode)
    rgb = np.array(img_f.convert("RGB"), dtype=np.uint8)  # H×W×3
    alpha = np.array(img_f.getchannel("A"), dtype=np.uint8)
    H, W = rgb.shape[:2]

    # 3) 采样网格（step 默认等于 block）
    if step is None or step <= 0:
        step = block

    # 生成 block 列表（输入坐标）
    blocks = []
    anchors_xy = []
    half = block // 2
    for y in range(0, H, step):
        for x in range(0, W, step):
            if anchor == "center":
                x0 = max(0, x - half)
                y0 = max(0, y - half)
                x1 = min(W, x0 + block)
                y1 = min(H, y0 + block)
                # 若靠边导致块变小，尝试往回挪动保证宽高尽量 == block
                if x1 - x0 < block and x0 > 0:
                    x0 = max(0, x1 - block)
                if y1 - y0 < block and y0 > 0:
                    y0 = max(0, y1 - block)
            else:  # topleft
                x0, y0 = x, y
                x1 = min(W, x0 + block)
                y1 = min(H, y0 + block)
            if x0 >= x1 or y0 >= y1:
                continue
            blocks.append((x0, y0, x1, y1))
            anchors_xy.append((x, y))  # 采样点（输入坐标），用于映射到输出

    if not blocks:
        print("❌ 没有生成任何采样块，请检查 block/step 参数。", file=sys.stderr)
        sys.exit(1)

    # 4) 块平均色（alpha 加权）
    colors = block_mean_colors_alpha(rgb, alpha, blocks)  # (M,3)
    # 过滤平均 alpha 太低的块（可选）
    if alpha_threshold > 0:
        # 计算每块平均 alpha（重用同样的方法）
        A = alpha.astype(np.float64)
        I_A = _integral(A)
        meanA = []
        for (x0, y0, x1, y1) in blocks:
            sumA = _rect_sum(I_A, x0, y0, x1, y1)
            area = (x1 - x0) * (y1 - y0)
            meanA.append(sumA / max(1.0, area))
        meanA = np.array(meanA)  # 0..255
        valid_mask = meanA > float(alpha_threshold)
    else:
        valid_mask = np.ones((len(blocks),), dtype=bool)

    # 5) 最近邻（批量）
    idx_all = np.full((len(blocks),), -1, dtype=np.int32)
    if valid_mask.any():
        idx_all[valid_mask] = searcher.query(colors[valid_mask], random_factor=random_factor, k_random=random_k)

    # 6) 输出画布与放置（按比例映射，允许重叠）
    #    scale = emoji_size / block；输入坐标 (x,y) -> 输出坐标 (round(x*scale), round(y*scale))
    scale = float(emoji_size) / float(block)
    W_out = int(np.ceil(W * scale))
    H_out = int(np.ceil(H * scale))
    canvas = Image.new("RGBA", (W_out, H_out), (0, 0, 0, 0))

    for (x_in, y_in), idx in zip(anchors_xy, idx_all):
        if idx < 0:  # 被 alpha_threshold 过滤的块
            continue
        tx = int(round(x_in * scale))
        ty = int(round(y_in * scale))
        tile = tile_cache[int(idx)]
        canvas.paste(tile, (tx, ty), tile)  # 允许重叠，超边界 Pillow 会自动裁剪

    canvas.save(out_path)
    print(f"✅ 完成：{out_path}  输出尺寸：{W_out}×{H_out}  块数：{len(blocks)}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="按输入图块覆盖贴 emoji（允许重叠），生成马赛克图。")
    ap.add_argument("--dict", required=True, help="rgb888 字典（每行：HEX FILENAME）")
    ap.add_argument("--emoji-dir", required=True, help="emoji 图片目录（字典里的文件名相对该目录）")
    ap.add_argument("--input", required=True, help="输入图片路径")
    ap.add_argument("--out", required=True, help="输出 PNG 路径")

    # 卷积与取样
    ap.add_argument("--kernel", choices=["box", "gauss"], default="box", help="3×3 卷积核（默认 box）")
    ap.add_argument("--passes", type=int, default=2, help="卷积次数（默认 2）")
    ap.add_argument("--conv-mode", choices=["same", "valid"], default="same", help="卷积尺寸语义（same/valid）")
    ap.add_argument("--block", type=int, default=16, help="每个 tile 覆盖的输入块边长（像素），默认 16")
    ap.add_argument("--step", type=int, default=None, help="采样步幅（像素），默认等于 block；可小于 block 以产生重叠")
    ap.add_argument("--anchor", choices=["topleft", "center"], default="center",
                    help="块锚点（默认 center 更符合“周围覆盖”）")

    ap.add_argument("--random-factor", type=float, default=0.0,
                    help="随机因子 [0..1]，引入随机性以避免大面积单色块（默认 0）")
    ap.add_argument("--random-k", type=int, default=10,
                    help="随机选择时考虑的最近邻数量（默认 10）")

    # 贴图尺寸
    ap.add_argument("--emoji-size", type=int, default=32, help="输出中每个 emoji tile 的像素尺寸（默认 32）")
    ap.add_argument("--alpha-threshold", type=int, default=0, help="块平均 alpha ≤ 阈值则跳过贴图（0..255）")

    args = ap.parse_args()

    process(
        dict_path=args.dict,
        emoji_dir=args.emoji_dir,
        input_path=args.input,
        out_path=args.out,
        kernel=args.kernel,
        passes=args.passes,
        conv_mode=args.conv_mode,
        block=args.block,
        step=args.step,
        emoji_size=args.emoji_size,
        anchor=args.anchor,
        alpha_threshold=args.alpha_threshold,
        random_factor=args.random_factor,
        random_k=args.random_k,
    )


if __name__ == "__main__":
    main()
