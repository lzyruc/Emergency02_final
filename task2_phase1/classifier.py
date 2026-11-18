"""4 类物体的规则分类 (基于 footprint + 高度 稳定量)

三个稳定尺寸: (footprint_long, footprint_short, height_abs)
- 排序后 (d1 >= d2 >= d3)
- 对物体真实规格:
  - 纸箱 500×350×300 -> d1=500, d2=350, d3=300
  - 椅子 850 高 + 550 直径 -> d1=850 (高) 或 550, d3=550 或 300
  - 垃圾桶 275×255×255 -> d1=275, d2=d3=255
  - 文件筐 308×303×257 -> d1=308, d2=303, d3=257
"""
from . import config


def _tri(x, lo, best, hi):
    """三角隶属度: x 在 best 处 =1, 到 lo/hi 处 =0, 外面 =0"""
    if x <= lo or x >= hi:
        return 0.0
    if x <= best:
        return (x - lo) / max(best - lo, 1e-6)
    return (hi - x) / max(hi - best, 1e-6)


def basket_size_score(fp_long, fp_short, h_abs):
    """v13.3: 一个 0~1 的"像不像整只文件筐"分数 (规格 308×303×257mm).

    为什么要它: 同一只筐常被 DBSCAN 切成好几块, 每块以前都返回固定 0.95,
    最终 top-1 靠"谁排在前面"来决定 —— 结果经常选到只有半截高的碎片,
    抓取线就画到了筐的侧面而不是顶棱 (113017_133 就是这样).
    改成连续分数后, 尺寸最完整的那块自然胜出.
    高度权重最大, 因为"高度够不够"最能说明这块点云有没有包含顶棱.
    """
    s_long = _tri(fp_long, 0.16, 0.31, 0.52)
    s_short = _tri(fp_short, 0.08, 0.30, 0.42)
    s_h = _tri(h_abs, 0.15, 0.30, 0.42)
    # 高度权重给到 0.60: 筐放在地上, 顶沿必然在 0.26~0.36m.
    # 只有半截高 (0.15~0.20m) 的碎片, 抓取线一定画不到顶棱上.
    return 0.25 * s_long + 0.15 * s_short + 0.60 * s_h


def classify(cluster):
    d1, d2, d3 = cluster["dims"]
    h_abs = cluster["height_abs"]
    h_bottom = cluster.get("height_bottom", 0.0)
    h_span = h_abs - h_bottom
    footprint_long = cluster["footprint_long"]
    footprint_short = cluster["footprint_short"]
    is_round = cluster["footprint_is_round"]
    n_pts = cluster["n_points"]

    # 颜色分类 (可能没有)
    color_info = cluster.get("color")
    color_cat = color_info["category"] if color_info else "unknown"
    # v11.7: 蓝色亚类 (vivid = 亮蓝 = 筐; muted = 灰蓝 = 桶/阴影筐)
    blue_intensity = color_info.get("blue_intensity") if color_info else None

    color_str = color_cat
    if color_cat == "blue" and blue_intensity:
        color_str = f"{blue_intensity}_blue"

    reasons = [f"dims={d1:.2f}/{d2:.2f}/{d3:.2f}",
               f"fp={footprint_long:.2f}x{footprint_short:.2f}",
               f"h={h_abs:.2f}",
               f"span={h_span:.2f}",
               f"round={is_round}", f"n={n_pts}",
               f"color={color_str}"]

    # ---- 1. 椅子: 高度 >= 65cm 且 4 项都过 ----
    if h_abs >= config.CHAIR_MIN_HEIGHT:
        if h_span < config.CHAIR_MIN_HEIGHT_SPAN:
            return "unknown", 0.3, "tall_flat_top(likely_table); " + "; ".join(reasons)
        # 椅子腿常被 DBSCAN 漏 (细而分散), 允许 h_bottom 到 45cm (座椅高度)
        if h_bottom > 0.45:
            return "unknown", 0.3, f"tall_but_no_floor_contact(hbot={h_bottom:.2f}); " + "; ".join(reasons)
        # 短边要求 (排除细长条家具边)
        if footprint_short < config.CHAIR_MIN_FOOTPRINT_SHORT:
            return "unknown", 0.3, f"tall_but_thin(fs={footprint_short:.2f}); " + "; ".join(reasons)
        # 长边上限 (排除大桌子被认成 chair)
        if footprint_long > config.CHAIR_MAX_FOOTPRINT_LONG:
            return "unknown", 0.3, f"tall_but_too_wide(fl={footprint_long:.2f},likely_table); " + "; ".join(reasons)
        # 颜色: 真椅子物理上就是黑办公椅
        # dark = 近距/正光, gray = 远距/背光/banding 让黑变浅 (不是真灰)
        # 两者都当有效证据, 差距缩小 (v11.6.1: gray 0.75→0.85, unknown 0.6→0.75)
        if color_cat in ("dark", "gray"):
            conf = 0.9 if color_cat == "dark" else 0.85
            return "chair", conf, "chair_by_height_span; " + "; ".join(reasons)
        if color_cat == "unknown":
            # 拿不到颜色但高度+跨度+尺寸都对 → 高置信椅子 (h_abs>0.65 已经很硬)
            return "chair", 0.75, "chair_no_color; " + "; ".join(reasons)
        return "unknown", 0.3, f"tall_but_wrong_color({color_cat}); " + "; ".join(reasons)

    # ==== 2 & 3: 蓝色物体 (垃圾桶 or 文件筐) ====
    # v11.7 用蓝色亚类区分:
    #   vivid_blue (S≥120 或 B-R≥45): 亮宝蓝 = 筐 (强证据)
    #   muted_blue (灰蓝): 桶更可能, 但阴影里的筐也会掉进这里, 靠尺寸兜
    #   规格: trash 27.5×25.5×25.5, basket 30.8×30.3×25.7 (差 3cm)
    max_dim = max(d1, d2, d3)
    min_dim = min(d1, d2, d3)
    aspect_3d = max_dim / max(min_dim, 1e-6)
    fp_aspect = footprint_long / max(footprint_short, 1e-6)

    # v13.3: 筐的置信度改成"颜色底分 + 尺寸吻合度", 不再是死的 0.95.
    #        这样同一只筐的多个碎片之间能分出高下, top-1 会选到最完整的那块.
    bsz = basket_size_score(footprint_long, footprint_short, h_abs)
    b_minus_r = color_info.get("b_minus_r", 0.0) if color_info else 0.0

    # v13.3: 灰蓝色 (B-R 偏低) 优先当垃圾桶处理.
    #   img_head 26 帧实测: 真文件筐 B-R = 53~70, 灰蓝垃圾桶 B-R = 22~45, 中间干净.
    #   以前 muted 那几条规则会把桶按尺寸捞成筐 (0.31×0.24 刚好卡进筐的窗口),
    #   于是抓取线画到了桶沿上. 现在先用颜色把桶挡在筐的所有规则之前.
    if (color_cat == "blue" and blue_intensity == "muted"
            and b_minus_r < config.BASKET_MIN_BMR):
        # 例外: 两个水平方向都明显大过圆桶 = 还是当筐 (可能是重阴影下的筐)
        big_enough = (footprint_long >= config.BASKET_BIG_FP_LONG
                      and footprint_short >= config.BASKET_BIG_FP_SHORT
                      and 0.20 <= h_abs <= 0.42)
        if not big_enough:
            fl_lo, fl_hi = config.TRASH_GRAY_FP_RANGE
            h_lo, h_hi = config.TRASH_GRAY_H_RANGE
            if fl_lo <= footprint_long <= fl_hi and h_lo <= h_abs <= h_hi:
                return "trash_can", 0.90, (f"trash_bluish_gray(bmr={b_minus_r:.0f}); "
                                           + "; ".join(reasons))
            return "unknown", 0.30, (f"bluish_gray_too_off(bmr={b_minus_r:.0f}); "
                                     + "; ".join(reasons))

    # (强 basket) 亮宝蓝色 → 100% 是筐, 不管尺寸小
    if color_cat == "blue" and blue_intensity == "vivid":
        # vivid + 尺寸大致合理就认筐
        if (0.10 <= footprint_long <= 0.60
                and 0.10 <= h_abs <= 0.50
                and footprint_short >= 0.08):
            conf = 0.86 + 0.13 * bsz          # 0.86 ~ 0.99
            return "basket", conf, f"basket_by_vivid_blue(sz={bsz:.2f}); " + "; ".join(reasons)

    # (强 trash) 灰蓝色 + 小 → 桶 (0.92)
    # fp_long≤0.28 = trash 独有区; ≥0.22 排碎片; h≥0.20 排侧躺筐一角
    if (color_cat == "blue" and blue_intensity == "muted"
            and 0.22 <= footprint_long <= 0.28
            and 0.20 <= h_abs <= 0.32
            and aspect_3d < 1.6):
        return "trash_can", 0.92, f"trash_muted_blue_cube(ar={aspect_3d:.2f}); " + "; ".join(reasons)

    # (中 basket) 灰蓝 + 尺寸像筐(≥0.30) → 阴影里的筐, 认筐 0.85
    if (color_cat == "blue" and blue_intensity == "muted"
            and 0.30 <= footprint_long <= 0.60
            and 0.15 <= h_abs <= 0.50
            and footprint_short >= 0.15):
        return "basket", 0.76 + 0.13 * bsz, f"basket_muted_shadowed(sz={bsz:.2f}); " + "; ".join(reasons)

    # (弱 trash) 老版兼容: color_info 里没 blue_intensity (旧数据) 时的兜底
    # 或没颜色但尺寸很像桶
    if (color_cat == "blue" and blue_intensity is None
            and 0.22 <= footprint_long <= 0.28
            and 0.20 <= h_abs <= 0.32
            and aspect_3d < 1.6):
        return "trash_can", 0.85, f"trash_blue_no_subtype(ar={aspect_3d:.2f}); " + "; ".join(reasons)

    # (兜底 basket) 蓝色 + 尺寸合理 → basket 0.90 (老规则, 保底捞回)
    in_basket_size = (config.BASKET_MIN_DIM1 <= footprint_long <= config.BASKET_MAX_DIM1
                      and 0.20 <= h_abs <= 0.40
                      and footprint_short >= 0.18)
    # v13.3: 兜底不能再是"只要蓝就行". 实测被它捞起来的全是垃圾桶碎片
    #        (fp 0.19×0.10 / 0.20×0.11). 真筐 308×303mm, 就算斜看+遮挡,
    #        fp_long 也有 0.24 以上. 收严到 0.24 / 0.12 / h∈[0.15,0.45].
    if (color_cat == "blue"
            and 0.24 <= footprint_long <= 0.60
            and 0.15 <= h_abs <= 0.45
            and footprint_short >= 0.12):
        return "basket", 0.74 + 0.13 * bsz, f"basket_by_blue(sz={bsz:.2f}); " + "; ".join(reasons)
    if in_basket_size and color_cat in ("blue", "other", "unknown"):
        return "basket", 0.8, "basket_cubic; " + "; ".join(reasons)

    # ---- 4. 纸箱: 长条形 或 立起, 棕色强证据 ----
    # 棕色 + 尺寸接近 box -> 高置信度 box
    if color_cat == "brown" and (footprint_long >= 0.30 or h_abs >= 0.30):
        if footprint_long <= 0.90:
            return "box", 0.9, "box_by_brown; " + "; ".join(reasons)

    if ((config.BOX_MIN_DIM1 <= footprint_long <= config.BOX_MAX_DIM1)
        or (config.BOX_MIN_DIM1 <= h_abs <= config.BOX_MAX_DIM1
            and footprint_long >= 0.28)):
        # 排除接近文件筐尺寸的
        if footprint_long > config.BASKET_MAX_DIM1 or h_abs > 0.40:
            # 颜色约束: 蓝色明显不对 (蓝筐)
            if color_cat == "blue":
                return "unknown", 0.3, "box_shape_but_blue; " + "; ".join(reasons)
            # 灰色/深色 不再直接拒绝 — 斜过来的纸箱采到灰色是正常的
            # 但短边太小 = 家具边而不是箱子
            if footprint_short < config.BOX_MIN_FOOTPRINT_SHORT:
                return "unknown", 0.3, f"box_shape_but_thin(fs={footprint_short:.2f}); " + "; ".join(reasons)
            # 长条太明显 = 桌腿/桌沿 (真箱 fp_aspect 最大 500/300=1.67)
            if fp_aspect > 1.9:
                return "unknown", 0.3, f"box_shape_but_elongated(fp_ar={fp_aspect:.2f},likely_furniture); " + "; ".join(reasons)
            # 灰色 = 中等置信度 (可能斜箱正面棕 + 侧面阴影)
            # 但要方一点 (fp_aspect < 1.7) 才认, 太长条 = 家具
            if color_cat == "gray":
                if fp_aspect < 1.7 and footprint_short >= 0.25:
                    return "box", 0.6, f"box_size(gray_possibly_angled,fp_ar={fp_aspect:.2f}); " + "; ".join(reasons)
                return "unknown", 0.3, f"box_shape_gray_but_shape_off(fp_ar={fp_aspect:.2f},fs={footprint_short:.2f}); " + "; ".join(reasons)
            # 深色 dark 不认成 box (椅子/桌下阴影)
            if color_cat == "dark":
                return "unknown", 0.3, "box_shape_but_dark(likely_chair); " + "; ".join(reasons)
            return "box", 0.8, "box_size; " + "; ".join(reasons)

    return "unknown", 0.2, "no_size_match; " + "; ".join(reasons)
