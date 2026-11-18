"""文件筐专用识别：颜色检测、深度验证、五板语义 Top-1 抓取边。

=== 为什么要推翻重来 ===
老流程是"深度聚类为主, 颜色只是事后加分":
    深度 -> 去地面 -> 体素 -> DBSCAN(eps=3.5cm) -> 一堆尺寸过滤 -> 分类 -> 阈值
文件筐是**蓝色塑料网格筐, 满身大洞**, 深度相机能直接透过洞看到地面. 结果就是:
  1. 筐的点云稀疏而且被洞切碎, DBSCAN 串不起来 -> 分裂成好几块;
  2. 每块都过不了 FILTER_MIN_POINTS=150 / FILTER_MIN_AREA=0.025 -> 全被扔掉
     (这就是用户那边 "raw=19 kept=0" 的直接原因);
  3. 侥幸活下来的往往是半截碎片 -> 抓取线画在筐腰上, 而且被离群点拉得又长又歪.
换句话说: **漏检和线歪是同一个病根 —— 用不可靠的深度去做"检测"这一步.**

新流程反过来:
    亮蓝 HSV 掩膜 (2D, 密实, 不受洞影响) = 检测      <- 决定"有没有、在哪"
    深度                                   = 验证+定位 <- 决定"是不是真筐、线在哪"
场地里只有文件筐是这种亮蓝 (垃圾桶是灰蓝, B-R 明显低), 所以颜色可以当主判据.
实测 28 帧: 蓝掩膜的最大连通域每一帧都正好是筐, 零误检.

=== 三层证据 ===
A. 颜色 (主): 双阈值(滞回)蓝掩膜, 强核 + 弱扩散, 阴影里的筐也能连上, 地面绝不会进来.
B. 尺寸 (验): 掩膜内的 3D 点(剔掉"透过网洞看到的地面"), 比对实物 308x303x257mm.
C. 结构 (证): 用户提的"正着放能看到五道棱" —— 沿主方向的高度剖面找峰;
              以及网洞造成的规则空洞. 两者都只加分, 不做卡关(卡关=漏检).

=== 抓取线 ===
检测部分继续使用上面的成熟颜色优先逻辑。抓取边改由
``basket_grasp_cv.extract_basket_grasps`` 决定，并严格采用用户红线语义：
  - 只允许左右外侧板和三块内隔板的自由上轮廓；
  - 五板可见时要求平行、等距和三维高度跨度同时成立；
  - L 侧板朝上时只从可见 L 自由链中选择一条完整直边，不补隐藏板；
  - 前后横梁、底边和孔洞边从不作为降级候选。
  - 五板/多段只作为内部候选，公开接口最多返回一条最高置信安全线。

旧的“任意最高外轮廓”实现只作为私有历史参考保留；公开
``best_grasp_line`` 已被安全禁用，正式 ``detect_baskets`` 只调用五板语义分支。
"""
import cv2
import numpy as np

from . import config
from .basket_grasp_cv import (
    ALGORITHM_VERSION as GRASP_ALGORITHM_VERSION,
    extract_basket_grasps,
)


# ============================================================
# 0. 小工具
# ============================================================

def _ground_frame(plane):
    """由地面方程造一个正交基: up(法向, 朝上) + 地面内两个方向 e1/e2"""
    up = np.asarray(plane[:3], dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-9)
    ref = np.array([0.0, 0.0, 1.0])          # 相机朝前
    e1 = np.cross(up, ref)
    if np.linalg.norm(e1) < 1e-6:            # 退化时换一个参考轴
        e1 = np.cross(up, np.array([1.0, 0.0, 0.0]))
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(up, e1)
    e2 /= max(np.linalg.norm(e2), 1e-9)
    return up, e1, e2


def _height(pts, plane):
    """点离地高度 (plane 法向已朝上, 所以直接是有符号距离)"""
    a, b, c, d = plane
    return a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d


def _unproject(u, v, z, intr):
    """像素+深度(m) -> 相机系 3D"""
    x = (np.asarray(u, dtype=np.float64) - intr["cx"]) * z / intr["fx"]
    y = (np.asarray(v, dtype=np.float64) - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=-1)


def _project(p, intr):
    z = max(float(p[2]), 1e-6)
    return np.array([p[0] * intr["fx"] / z + intr["cx"],
                     p[1] * intr["fy"] / z + intr["cy"]])


def _gauss(x, mu, sigma):
    return float(np.exp(-((x - mu) / max(sigma, 1e-6)) ** 2))


def _ngauss(x, mu, sigma):
    """标准正态形状的钟形函数 (带 1/2 系数). 和上面的 _gauss 差一个 sqrt(2) 的宽度;
    新的置信度公式用这个, 常数是照着它离线标定出来的, 不要混用."""
    return float(np.exp(-0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2))


def _ramp(x, lo, hi):
    """线性斜坡: <=lo 得 0, >=hi 得 1, 中间线性. 比硬阈值稳 —— 测量值在
    边界附近抖动时分数是连续变化的, 不会一票翻盘."""
    return float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def _ray_line_point(uv, intr, ctr, dvec):
    """求 3D 直线上离"过像素 uv 的视线"最近的那个点.

    为什么需要它: 抓取线的**长度和端点**必须由 2D 轮廓决定 (轮廓端点是精确的,
    而且天然落在物体上), 深度只负责给这条线定个位置. 如果反过来用"深度采样点的
    最远两端"当端点, 只要有一小段边没有有效深度 (筐是镂空的, 这太常见了),
    线就会缩短一大截 —— 实测出现过 0.107m 的线 (实物棱长 0.303m).
    """
    r = np.array([(uv[0] - intr["cx"]) / intr["fx"],
                  (uv[1] - intr["cy"]) / intr["fy"], 1.0])
    r = r / np.linalg.norm(r)
    d = dvec / max(np.linalg.norm(dvec), 1e-9)
    b = float(d @ r)
    denom = 1.0 - b * b
    if denom < 1e-6:                      # 视线和直线几乎平行, 退化
        return None
    dd = float(d @ ctr)
    e = float(r @ ctr)
    t = (b * e - dd) / denom
    return ctr + t * d


def _obb_2d(xy):
    """地面内的最小面积外接矩形 -> (长边, 短边).

    不能直接用"主方向上的跨度": 筐是个方盒子, 从斜角看过去, 主方向正好是**对角线**,
    量出来 0.43m 而实物边长只有 0.308m, 尺寸分因此一直偏低 (实测 0.37~0.58).
    minAreaRect 量的是真正的边长.
    """
    if len(xy) < 8:
        return 0.0, 0.0
    # 去掉离中心最远的一撮点再量.
    # 为什么要去掉这么多 (10%): 筐是**网状的**, 网眼里的那些像素测到的是筐后面的
    # 地面 (远 0.2~0.35m). 这些点仍然落在蓝掩膜里, 沿视线方向把点云拉长一大截,
    # minAreaRect 于是量出 0.42~0.47m (实物只有 0.308m). 实测把最外圈 10% 去掉,
    # 28 张里绝大多数回到 0.31~0.38m, 尺寸分才有意义.
    c = np.median(xy, axis=0)
    r = np.linalg.norm(xy - c, axis=1)
    keep = r <= np.percentile(r, config.BCV_OBB_TRIM_PCT)
    pts = xy[keep] if keep.sum() >= 8 else xy
    try:
        rect = cv2.minAreaRect((pts * 1000.0).astype(np.float32))
        w, h = rect[1]
        w, h = w / 1000.0, h / 1000.0
    except Exception:
        w = float(np.ptp(pts[:, 0]))
        h = float(np.ptp(pts[:, 1]))
    return float(max(w, h)), float(min(w, h))


# ============================================================
# 1. 蓝掩膜 (检测主判据)
# ============================================================

def _split_thin_necks(bin_mask):
    """把只靠一根**细脖子**连在一起的两团掩膜切开, 并给出等价于
    cv2.connectedComponentsWithStats 的 (n, lab, stats).

    === 为什么需要 ===
    补网洞的 close(BCV_CLOSE_PX=9) 是必须的, 否则一个筐会被网眼切成几十块;
    但它也会顺手把**图上只差几像素**的两个不同物体焊在一起. 124638_550:
    筐 (18310px) + 灰垃圾桶的桶内壁 (7201px, 背光时内壁也偏蓝) 被焊成一团
    25522px, 报出来的占地 0.90x0.39m —— 筐实际只有 0.308x0.303m.

    === 做法 (标准的"腐蚀-种子-长回去") ===
    1. 先腐蚀 r 像素. 脖子比两边的身子细得多, 所以**脖子先断**, 剩下两个种子.
    2. 让各个种子在原掩膜内同时一圈圈长回去, 先到先得 —— 于是分界线正好
       落在脖子最细处.
    3. 整个被腐蚀光的分量 (远处很小的筐) 没有种子, 原样保留, 绝不丢.
    实测 r=5/6/7 结果完全一样 (7201+18310), r=4 会留下 271px 的毛刺,
    所以取 6 留一点余量. 筐哪怕在 3m 外身子也有 40+px 宽, 腐蚀 6 不会切坏它.
    """
    m = (bin_mask > 0).astype(np.uint8)
    r = config.BCV_SPLIT_ERODE
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    seed = cv2.erode(m, k)
    ns, slab = cv2.connectedComponents(seed, 8)
    lab = slab.astype(np.float32)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(config.BCV_SPLIT_ITER):
        grown = cv2.dilate(lab, k3)          # 3x3 max 滤波 = 标签往外长一圈
        new = np.where((lab == 0) & (m > 0), grown, lab)
        if np.array_equal(new, lab):
            break
        lab = new
    lab = lab.astype(np.int32)
    # 一个种子都没沾到的分量 (整个被腐蚀掉了) -> 单独编号保留
    left = ((lab == 0) & (m > 0)).astype(np.uint8)
    if left.any():
        nl, llab = cv2.connectedComponents(left, 8)
        lab = np.where(left > 0, llab + (ns - 1), lab)
        ns = ns + nl - 1
    # 自己算 stats, 布局跟 OpenCV 一致: [LEFT, TOP, WIDTH, HEIGHT, AREA].
    # 用 ufunc.at 一次扫完, 不要 for i in range(ns) 里再 lab==i —— 那是
    # O(标签数 x 全图), 几百个标签就是几秒.
    ys, xs = np.nonzero(lab > 0)
    ids = lab[ys, xs]
    stats = np.zeros((ns, 5), dtype=np.int32)
    if ids.size:
        area = np.bincount(ids, minlength=ns)
        BIG = 1 << 30
        x0 = np.full(ns, BIG, dtype=np.int64)
        np.minimum.at(x0, ids, xs)
        y0 = np.full(ns, BIG, dtype=np.int64)
        np.minimum.at(y0, ids, ys)
        x1 = np.full(ns, -1, dtype=np.int64)
        np.maximum.at(x1, ids, xs)
        y1 = np.full(ns, -1, dtype=np.int64)
        np.maximum.at(y1, ids, ys)
        ok = area > 0
        stats[ok, 0] = x0[ok]
        stats[ok, 1] = y0[ok]
        stats[ok, 2] = x1[ok] - x0[ok] + 1
        stats[ok, 3] = y1[ok] - y0[ok] + 1
        stats[ok, 4] = area[ok]
    return ns, lab, stats


def blue_mask(rgb_bgr, debug=None):
    """滞回(双阈值)亮蓝掩膜.

    为什么要双阈值: 筐一半在阳光下一半在阴影里, 单阈值必然二选一 ——
    松了地面阴影混进来, 紧了阴影里的半个筐丢掉. 滞回的做法是:
      strong: 很确定是亮蓝的核心像素 (饱和度高, B-R 大)
      weak  : 放宽一大截的候选
    只保留**含有 strong 像素**的 weak 连通域 -> 阴影部分能顺着核心长出来,
    而完全没核心的区域 (地面反光、蓝灰垃圾桶) 一个像素都进不来.
    """
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    B = rgb_bgr[:, :, 0].astype(np.int16)
    R = rgb_bgr[:, :, 2].astype(np.int16)
    BmR = B - R

    strong = ((H >= config.BCV_H_LO) & (H <= config.BCV_H_HI) &
              (S >= config.BCV_S_STRONG) & (V >= config.BCV_V_STRONG) &
              (BmR >= config.BCV_BMR_STRONG)).astype(np.uint8)
    # 色度: (B-R)/V. 除掉亮度之后才是"有多蓝", 暗处的真筐照样过得去,
    # 而黑椅子、筐自己的阴影这种"暗且不太蓝"的会被挡在外面 (见 config 注释).
    rel = BmR.astype(np.float32) / np.maximum(V, 1).astype(np.float32)
    weak = ((H >= config.BCV_H_LO - 4) & (H <= config.BCV_H_HI + 4) &
            (S >= config.BCV_S_WEAK) & (V >= config.BCV_V_WEAK) &
            (BmR >= config.BCV_BMR_WEAK) &
            (rel >= config.BCV_REL_WEAK)).astype(np.uint8)
    # 滞回的前提是 strong 必须是 weak 的子集 (否则下面"数每个连通域里有多少
    # strong 像素"会把落在背景上的 strong 算丢). 加了 rel 关之后, 极亮的
    # 核心像素理论上可能被卡掉 (B-R=40 而 V=160 -> rel=0.25), 所以显式并回去.
    weak = np.maximum(weak, strong)

    # 先把网洞补上再做连通域, 否则一个筐会被洞切成几十块.
    # 洞的直径实测 10~18px, 用 close 半径 = BCV_CLOSE_PX 的椭圆核一次补掉.
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.BCV_CLOSE_PX, config.BCV_CLOSE_PX))
    weak_c = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, k)
    # 补完再开一次, 去掉零星噪点 (地面上的蓝色反光小点)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    weak_c = cv2.morphologyEx(weak_c, cv2.MORPH_OPEN, k2)

    # 补洞的副作用: 会把**本来分开**的两团焊在一起. 124638_550 就是这样 ——
    # 筐和它后面那个灰垃圾桶的**桶内壁**(背光时也发蓝)在图上只差几个像素,
    # close(9) 一下就连成了一个 25522px 的大团, 于是报出来的占地是
    # 0.90x0.39m, 而筐只有 0.31m. 所以补完洞再按"细颈"把它切开.
    n, lab, stats = _split_thin_necks(weak_c)
    keep = np.zeros(n, dtype=bool)
    if n > 1:
        # 每个 label 里 strong 像素数
        s_idx = lab[strong.astype(bool)]
        if s_idx.size:
            cnt = np.bincount(s_idx, minlength=n)
        else:
            cnt = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= config.BCV_MIN_AREA_PX and \
               cnt[i] >= config.BCV_MIN_STRONG_PX:
                keep[i] = True
    out = keep[lab].astype(np.uint8)
    if debug is not None:
        debug["n_strong"] = int(strong.sum())
        debug["n_weak"] = int(weak.sum())
        debug["n_kept"] = int(out.sum())
    return out, lab, stats, keep, (H, S, V, BmR)


# ============================================================
# 2. blob -> 3D 点 (剔掉透过网洞看到的地面)
# ============================================================

def _z_ok(z, z_near):
    """深度窗口: 锚在物体的**近端**, 只往远处放宽.

    往近处只留 6cm 的容差 (噪声), 往远处放到 BCV_Z_SPAN (比筐的对角线 0.43m
    还宽一点). 物理上: 沿这条视线上, 蓝色的东西里最近的那层就是筐的前表面,
    不可能有别的蓝东西挡在它前面; 而"更远"的读数既可能是筐自己的后半部分,
    也可能是透过网眼看到的地面 —— 后者由 h > BCV_MIN_H 那道门去掉.
    """
    return (z > z_near - config.BCV_Z_NEAR_TOL) & (z < z_near + config.BCV_Z_SPAN)


def _blob_points(mask_blob, depth_mm, intr, plane):
    """取掩膜内的 3D 点, 并剔掉"透过网洞看到的地面/远处背景".

    深度参考点用**近端** z_near (第 12 分位), 不是中位数 z_med.
    为什么: 筐是网状的, 从某些角度看过去, 蓝掩膜里**大半的像素测到的是筐后面的
    地面**(实测 201127_477: 掩膜内 z 的中位数 1.421m, 而筐的近侧沿只有 1.18m).
    这时 z_med 落在"筐后面的地面"上, 以它为中心 ±0.30m 开窗, 会把最前面那条
    **真筐沿**切掉一大半 —— 顶面那一层 1729px 里有 1080px 是被这一刀砍掉的.
    改成锚在近端 + 单侧放宽 (筐自己的对角线才 0.43m) 就不会砍到自己了.
    """
    vs, us = np.nonzero(mask_blob)
    if len(us) < 30:
        return None
    z_mm = depth_mm[vs, us]
    ok = (z_mm > 0) & (z_mm < config.MAX_DEPTH_MM)
    if ok.sum() < 30:
        return None
    us, vs, z = us[ok], vs[ok], z_mm[ok].astype(np.float64) / 1000.0
    z_med = float(np.median(z))
    z_near = float(np.percentile(z, config.BCV_Z_NEAR_PCT))
    pts = _unproject(us, vs, z, intr)
    h = _height(pts, plane)
    inband = _z_ok(z, z_near)
    sel = inband & (h > config.BCV_MIN_H)
    if sel.sum() < 30:
        # 退一步: 只做深度带过滤 (物体可能很矮/贴地)
        sel = inband
        if sel.sum() < 30:
            return None
    return {"pts": pts[sel], "h": h[sel],
            "uv": np.stack([us[sel], vs[sel]], axis=1),
            "z_med": z_med, "z_near": z_near,
            "n_valid": int(ok.sum()), "n_mask": int(mask_blob.sum())}


# ============================================================
# 3. 结构先验: "正着放能看到五道棱" + 网洞签名
# ============================================================

def _ridge_score(pts, h, plane):
    """沿地面内主方向做高度剖面, 数棱.

    用户的先验: 文件筐正面朝着相机时, 4 个格子 5 道隔板棱非常明显.
    这里把点投到地面内, 找主方向, 每 1.5cm 一个格子取该格最高点,
    平滑后找显著的峰. 真筐 (露隔板那面) 会出 3~6 个峰, 间距 5~9cm 且比较均匀;
    纸箱/垃圾桶/桌子的剖面是平的, 一个峰都没有.

    返回 (score 0~1, n_peak, 中位间距m)
    """
    if len(pts) < 80:
        return 0.0, 0, 0.0
    up, e1, e2 = _ground_frame(plane)
    a = pts @ e1
    b = pts @ e2
    A = np.stack([a - a.mean(), b - b.mean()], axis=1)
    # 地面内的主方向
    C = A.T @ A
    w, V = np.linalg.eigh(C)
    d1 = V[:, np.argmax(w)]
    t = A @ d1
    lo, hi = t.min(), t.max()
    if hi - lo < 0.08:
        return 0.0, 0, 0.0
    nb = max(8, int(np.ceil((hi - lo) / config.BCV_RIDGE_BIN)))
    idx = np.clip(((t - lo) / (hi - lo) * (nb - 1)).astype(int), 0, nb - 1)
    prof = np.full(nb, np.nan)
    for i in range(nb):
        m = idx == i
        if m.sum() >= 3:
            prof[i] = np.percentile(h[m], 95)
    good = ~np.isnan(prof)
    if good.sum() < 8:
        return 0.0, 0, 0.0
    # 线性插值补空格 + 3 点平滑
    xs = np.arange(nb)
    prof = np.interp(xs, xs[good], prof[good])
    ker = np.ones(3) / 3.0
    sm = np.convolve(prof, ker, mode='same')
    # 找峰: 比左右都高, 且相对两侧最近谷的落差 (prominence) 够大
    peaks = []
    for i in range(1, nb - 1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]:
            L = sm[max(0, i - 6):i].min() if i > 0 else sm[i]
            Rr = sm[i + 1:min(nb, i + 7)].min() if i < nb - 1 else sm[i]
            prom = sm[i] - max(L, Rr)
            if prom >= config.BCV_RIDGE_PROM:
                peaks.append(i)
    if len(peaks) < 2:
        return 0.0, len(peaks), 0.0
    step = (hi - lo) / max(nb - 1, 1)
    gaps = np.diff(peaks) * step
    med = float(np.median(gaps))
    n = len(peaks)
    # 数量: 3~6 个最像 (5 道棱, 侧看可能少一两道)
    s_n = 1.0 if 3 <= n <= 6 else (0.55 if n == 2 or n == 7 else 0.2)
    # 间距: 筐宽 0.303m / 4 格 ~= 0.076m
    s_gap = _gauss(med, 0.076, 0.035)
    # 均匀度
    cv_ = float(np.std(gaps) / max(med, 1e-6))
    s_uni = float(np.clip(1.0 - cv_ / 0.6, 0.0, 1.0))
    return float(np.clip(0.45 * s_n + 0.35 * s_gap + 0.20 * s_uni, 0, 1)), n, med


def _hole_score(mask_blob, blue_raw):
    """网洞签名: 填充后的轮廓内, 有多少比例不是蓝的 = 洞.

    文件筐是网格的, 露网那面洞率能到 0.2~0.5; 实心那面接近 0.
    纸箱/垃圾桶/桌子根本没有这种规则空洞. 只加分不卡关.
    """
    cnts, _ = cv2.findContours(mask_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(mask_blob)
    cv2.drawContours(filled, [c], -1, 1, -1)
    area = float(filled.sum())
    if area < 200:
        return 0.0, 0.0
    inside_blue = float((filled.astype(bool) & blue_raw.astype(bool)).sum())
    hole_ratio = 1.0 - inside_blue / area
    # 0 -> 0 分, 0.25 附近最典型, 0.6 以上说明掩膜碎了不加分
    s = _gauss(hole_ratio, 0.25, 0.22) if hole_ratio < 0.6 else 0.2
    return float(np.clip(s, 0, 1)), float(hole_ratio)


# ============================================================
# 4. 抓取线: 2D 轮廓定线 -> 3D 抬升 -> 打分
# ============================================================

def _poly_segments(ap, out):
    """在一个闭合多边形的顶点环上枚举候选直边 (含跨顶点合并)."""
    n = len(ap)
    if n < 2:
        return
    for i in range(n):
        for span in range(1, min(config.BCV_MAX_SPAN, n - 1) + 1):
            j = (i + span) % n
            P, Q = ap[i], ap[j]
            L = np.linalg.norm(Q - P)
            if L < config.BCV_MIN_SEG_PX:
                continue
            # 跨多个角点时, 中间角点必须几乎在弦上, 否则不是一条直边
            if span > 1:
                dv = (Q - P) / max(L, 1e-9)
                nv = np.array([-dv[1], dv[0]])
                ok = True
                for s in range(1, span):
                    M = ap[(i + s) % n]
                    if abs(float((M - P) @ nv)) > config.BCV_COLLINEAR_PX:
                        ok = False
                        break
                if not ok:
                    continue
            out.append((P, Q))


def _contour_segments(mask_blob):
    """从掩膜轮廓上取候选直边.

    两个来源, 取并集:
      (a) 原轮廓的 approxPolyDP 角点 —— 保留凹处的真实结构 (比如筐口的缺口);
      (b) **凸包**的 approxPolyDP 角点 —— 这是这一版新加的关键.
          网格筐的侧影全是锯齿(网眼咬边), 直接在原轮廓上取角点, 一条完整的顶沿
          会被咬成 3~4 段, 每段都只有 50~90px -> 抬到 3D 就是 0.10~0.15m 的碎线.
          凸包把凹进去的锯齿全抹平, 顶沿变成一条完整的包边.
    弦有可能横跨一个大凹口(= 画在空气里), 所以后面 _refit_2d 会算"覆盖率"卡掉它.
    """
    cnts, _ = cv2.findContours(mask_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, []
    c = max(cnts, key=cv2.contourArea)
    return _one_contour_segments(c)


def _one_contour_segments(c):
    """对单条闭合轮廓做上面那件事. (顶面那一层可能有好几块, 要分别处理)"""
    if len(c) < 20:
        return None, []
    peri = cv2.arcLength(c, True)
    raw = []
    ap = cv2.approxPolyDP(c, config.BCV_APPROX_EPS * peri,
                          True).reshape(-1, 2).astype(np.float64)
    _poly_segments(ap, raw)
    try:
        hull = cv2.convexHull(c)
        hp = cv2.approxPolyDP(hull, config.BCV_HULL_EPS_PX,
                              True).reshape(-1, 2).astype(np.float64)
        _poly_segments(hp, raw)
    except Exception:
        pass
    # 去重 (两个来源会给出几乎一样的边)
    segs, seen = [], set()
    for (P, Q) in raw:
        k1 = (int(P[0] // 4), int(P[1] // 4), int(Q[0] // 4), int(Q[1] // 4))
        k2 = (k1[2], k1[3], k1[0], k1[1])
        if k1 in seen or k2 in seen:
            continue
        seen.add(k1)
        segs.append((P, Q))
    return c.reshape(-1, 2).astype(np.float64), segs


def _height_image(depth_mm, intr, plane):
    """整幅图的"离地高度"图 + 有效掩膜 + 深度(m). 用来找"最高的那一层面"."""
    h_img, w_img = depth_mm.shape[:2]
    z = depth_mm.astype(np.float64) / 1000.0
    ok = (z > 0.2) & (z < config.MAX_DEPTH_MM / 1000.0)
    vv, uu = np.mgrid[0:h_img, 0:w_img]
    X = (uu - intr["cx"]) * z / intr["fx"]
    Y = (vv - intr["cy"]) * z / intr["fy"]
    a, b, c, d = plane
    return a * X + b * Y + c * z + d, ok, z


def _top_band_mask(mask_blob, hmap, okmap, zmap, z_ref, h_top):
    """物体**最高的那一层面**的像素掩膜.

    === 为什么必须加这一步 (这是这一版最关键的修正) ===
    原来所有候选边都只从"蓝掩膜的外轮廓"上取. 但实拍图里有两种情况, 真正该抓的
    那条棱**根本不在外轮廓上**:

    (1) 筐倒扣着放 (底朝天). 最高的是那块实心底板, 它的边在侧影**里面**;
        侧影的上边界是四周支棱出来的塑料齿 —— 那些齿是镂空的, 深度直接穿过去
        打在后面的地面上, 所以沿着齿画出来的线一律算成 h~0.02m (贴地), 得分极低.
    (2) 筐正着放、从上往下看. 近侧的筐沿是一条又宽又亮的高台 (h~0.34), 但远侧
        看到的是筐**内壁和筐里的地面** (h~0), 侧影的上边界就落在那儿.

    换句话说: 侧影边界 != 物体最高的那条棱. 所以这里直接按高度切一层
    "顶面" 出来, 再从这层顶面的轮廓上取候选边 —— 那才是夹爪真能咬住的地方.
    """
    band = ((mask_blob > 0) & okmap & _z_ok(zmap, z_ref) &
            (hmap >= h_top - config.BCV_TOP_BAND)).astype(np.uint8)
    if band.sum() < config.BCV_TOP_MIN_AREA:
        return None
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, k5)
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k3)
    return band if band.sum() >= config.BCV_TOP_MIN_AREA else None


def _band_ransac(band, max_lines=2, seed=12345):
    """在**整块顶面带**上直接做 RANSAC 共线拟合, 不管它被断成了几块.

    === 为什么必须再加这一路 ===
    顶面带常常是**断开的**. 201127_477 是最典型的例子: 筐正着放, 近侧那条筐沿
    在高度图里是一条又高又亮的带 (h~0.28), 但网眼反光把它打断成 449px + 322px
    两截, 中间还夹着几块隔板棱的碎片. 按连通域各自取轮廓, 每一截只有
    0.10~0.18m, 长度分极低, 于是输给了一条 0.275m 的侧壁竖边 —— 可这两截本来
    就是**同一条 0.30m 的筐沿**.

    RANSAC 不问连通性, 只问"哪些像素共线", 所以一次就能把整条沿拼回来.
    做法: 随机取两点定一条线 -> 数容差内的内点 -> 取内点最多的那条 -> 用内点
    重新做 PCA 拟合 -> 端点取内点在主方向上的两极. 拿掉内点再找第二条.

    返回 [(P, Q), ...], 至多 max_lines 条.
    """
    vs, us = np.nonzero(band)
    if len(us) < config.BCV_RANSAC_MIN_PTS:
        return []
    pts = np.stack([us, vs], axis=1).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = []
    alive = np.ones(len(pts), dtype=bool)
    for _ in range(max_lines):
        cur = pts[alive]
        if len(cur) < config.BCV_RANSAC_MIN_PTS:
            break
        best_in = None
        n_iter = config.BCV_RANSAC_ITER
        ia = rng.integers(0, len(cur), n_iter)
        ib = rng.integers(0, len(cur), n_iter)
        for a, b in zip(ia, ib):
            d = cur[b] - cur[a]
            L = np.linalg.norm(d)
            if L < config.BCV_RANSAC_MIN_BASE:
                continue                      # 两点太近, 方向不可靠
            d = d / L
            nv = np.array([-d[1], d[0]])
            s = np.abs((cur - cur[a]) @ nv)
            inl = s < config.BCV_RANSAC_TOL
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
        if best_in is None:
            break
        n_in = int(best_in.sum())
        if n_in < config.BCV_RANSAC_MIN_PTS or n_in < 0.15 * len(pts):
            break
        sub = cur[best_in]
        ctr = sub.mean(axis=0)
        X = sub - ctr
        w, V = np.linalg.eigh(X.T @ X)
        dd = V[:, int(np.argmax(w))]
        t = X @ dd
        out.append((ctr + t.min() * dd, ctr + t.max() * dd))
        # 把用掉的内点从候选池里划掉, 再找下一条
        idx = np.nonzero(alive)[0][best_in]
        alive[idx] = False
    return out


def _refit_2d(contour_xy, P, Q):
    """把候选边贴回轮廓: 取靠近这条弦的轮廓点做总体最小二乘, 重定端点.

    approxPolyDP 是角点到角点, 边上如果有点毛刺, 弦会稍微偏离真实棱线.
    这里用弦附近 BCV_BAND_PX 内的轮廓点重新拟合一条直线, 再把端点投影上去,
    线就精确落在棱的"平均位置"上 —— 这是"贴合"的关键一步.

    同时算两个"这条线是不是真的贴着实体"的指标 (把线等分 24 段来数):
      - **覆盖率 cover**: 有多少段真的有轮廓点贴着;
      - **最大空档 maxgap**: 连续没有轮廓点的段, 最长的一串占全长的比例.
    为什么要两个: 筐口一圈是**锯齿状的凸起**(网格的齿), 凸包的弦只碰到齿尖,
    覆盖率天然只有 0.3~0.5 —— 但这是一条**真的顶沿**. 而"横跨一个大凹口"的
    假弦 (凭空画在空气里) 的特征是**中间一大段连续空白**. 所以真正该卡的是
    maxgap, cover 只做个很松的底线.
    返回 (A, B, 平均偏差px, 参与点数, 覆盖率, 最大空档)
    """
    d = Q - P
    L = np.linalg.norm(d)
    if L < 1e-6:
        return P, Q, 999.0, 0, 0.0, 1.0
    d = d / L
    nv = np.array([-d[1], d[0]])
    rel = contour_xy - P
    t = rel @ d
    s = np.abs(rel @ nv)
    sel = (t > -2) & (t < L + 2) & (s < config.BCV_BAND_PX)
    pts = contour_xy[sel]
    if len(pts) < 8:
        return (P, Q, float(s[sel].mean()) if sel.any() else 999.0,
                int(sel.sum()), 0.0, 1.0)
    ctr = pts.mean(axis=0)
    X = pts - ctr
    w, V = np.linalg.eigh(X.T @ X)
    dd = V[:, int(np.argmax(w))]
    if dd @ d < 0:
        dd = -dd
    nn = np.array([-dd[1], dd[0]])
    dev = float(np.mean(np.abs(X @ nn)))
    tt = X @ dd
    lo, hi = float(tt.min()), float(tt.max())
    nb = 24
    if hi - lo > 1e-6:
        tb = np.clip(((tt - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
        occ = np.zeros(nb, dtype=bool)
        occ[np.unique(tb)] = True
        cover = float(occ.sum()) / nb
        run = best = 0
        for f in occ:
            run = 0 if f else run + 1
            best = max(best, run)
        maxgap = float(best) / nb
    else:
        cover, maxgap = 0.0, 1.0
    A = ctr + lo * dd
    Bp = ctr + hi * dd
    return A, Bp, dev, int(len(pts)), cover, maxgap


def _outward_normal(mask_blob, P, Q):
    """判断哪一侧是物体外面 (掩膜像素少的那侧)"""
    d = Q - P
    L = np.linalg.norm(d)
    if L < 1e-6:
        return np.array([0.0, -1.0])
    d = d / L
    nv = np.array([-d[1], d[0]])
    h, w = mask_blob.shape
    ts = np.linspace(0.1, 0.9, 15)
    cnt = [0, 0]
    for si, sgn in enumerate((1.0, -1.0)):
        for t in ts:
            base = P + t * L * d + sgn * nv * 6.0
            u, v = int(round(base[0])), int(round(base[1]))
            if 0 <= u < w and 0 <= v < h and mask_blob[v, u]:
                cnt[si] += 1
    return nv if cnt[0] < cnt[1] else -nv


def _lift_segment(P, Q, out_n, depth_mm, intr, plane, mask_blob, z_ref):
    """沿 2D 线段采样深度 -> 3D 点.

    三个关键点:
    1) 深度只在**线内侧** (物体那一侧) 取: 线正上方就是背景, 直接取会拿到地面.
    2) **所有采样必须锚定在物体自己的深度 z_ref 上** (= 蓝掩膜内的中位深度).
       筐是镂空的, 5x5 窗口里经常一大半是"透过网洞看到的地面", 如果只按"这一段
       自己的中位深度"过滤, 整条边可能集体锁到地面上去 —— 实测就是这么出的
       h=0.039m 的抓取线 (画到了筐和地面的接触线上). 用 z_ref 锚定后, 地面
       (通常比筐远 0.3m 以上) 从一开始就进不来.
    3) **一个像素窗口里要取"最近的那层面", 不是中位数.**
       物理上: 我们采的是轮廓往里 4px 的点, 这一条视线上不可能有别的东西挡在
       筐前面 —— 筐自己就是最近的那层. 而"更远"的读数只有两个来源: 透过网眼
       看到的地面、以及**从上往下看时筐口里面的地面**. 后者是筐直立时顶沿被
       算成 h=0.04m (贴地!) 的真凶: 顶沿往里 4px 正好是筐口, 一眼看到筐内的
       地面, 它和 z_ref 只差 0.2m 出头, 混得过 BCV_Z_BAND. 改取低分位数
       (最近的 25%) 之后, 顶沿就锁在塑料边框上了.
    4) 收尾那一遍不能拿"常数中位深度"卡: 一条 0.3m 的棱斜着往远处走, 两端深度
       本来就能差 0.2m. 改成对 z(t) 拟合一条直线再剔离群点.
    """
    h_img, w_img = depth_mm.shape[:2]
    d = Q - P
    L2 = np.linalg.norm(d)
    if L2 < 1e-6:
        return None
    d = d / L2
    inw = -out_n
    N = max(12, min(80, int(L2 / 3)))
    ts = np.linspace(0.03, 0.97, N)
    zs, us, vs, tt = [], [], [], []
    for t in ts:
        base = P + t * L2 * d + inw * config.BCV_INWARD_PX
        u0, v0 = int(round(base[0])), int(round(base[1]))
        if not (2 <= u0 < w_img - 2 and 2 <= v0 < h_img - 2):
            continue
        patch = depth_mm[v0 - 2:v0 + 3, u0 - 2:u0 + 3].astype(np.float64)
        pv = patch[(patch > 0) & (patch < config.MAX_DEPTH_MM)] / 1000.0
        if pv.size < 3:
            continue
        # 锚定: 只认落在"物体近端往后一个筐深"这个窗口里的读数
        pv = pv[_z_ok(pv, z_ref)]
        if pv.size < 2:
            continue
        # 取最近的那层面 (低分位数), 见上面第 3 点
        zs.append(float(np.percentile(pv, config.BCV_NEAR_PCT)))
        us.append(base[0])
        vs.append(base[1])
        tt.append(t * L2)
    if len(zs) < 8:
        return None
    zs = np.asarray(zs)
    us = np.asarray(us)
    vs = np.asarray(vs)
    tt = np.asarray(tt)
    # 第二遍: 沿这条边把 z(t) 拟合成一条直线, 再剔掉偏离它的飞点.
    # (不能用常数中位数: 斜着往远处走的棱两端深度差得很正常)
    keep = np.ones(len(zs), dtype=bool)
    for _ in range(2):
        A = np.stack([tt[keep], np.ones(int(keep.sum()))], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, zs[keep], rcond=None)
        except np.linalg.LinAlgError:
            break
        res = np.abs(zs - (coef[0] * tt + coef[1]))
        nk = res < config.BCV_LINE_Z_BAND
        if nk.sum() < 8:
            break
        keep = nk
    if keep.sum() < 8:
        return None
    zs, us, vs = zs[keep], us[keep], vs[keep]
    pts = _unproject(us, vs, zs, intr)
    return {"pts": pts, "uv": np.stack([us, vs], axis=1), "n": len(zs),
            "valid_frac": float(len(zs)) / max(N, 1)}


def _fit_line3d(pts, trim=0.15):
    """3D 直线拟合 + 去掉最差的一小撮点再拟合. 返回 (ctr, dir, rms, 保留掩膜)"""
    keep = np.ones(len(pts), dtype=bool)
    ctr = dvec = None
    rms = 0.0
    for _ in range(2):
        P = pts[keep]
        ctr = P.mean(axis=0)
        X = P - ctr
        w, V = np.linalg.eigh(X.T @ X)
        dvec = V[:, int(np.argmax(w))]
        res = np.linalg.norm((pts - ctr) - np.outer((pts - ctr) @ dvec, dvec), axis=1)
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
        thr = np.quantile(res[keep], 1.0 - trim)
        nk = keep & (res <= max(thr, 1e-6))
        if nk.sum() < 8:
            break
        keep = nk
    return ctr, dvec, rms, keep


def _free_side(P, Q, out_n, depth_mm, mask_blob, z_edge):
    """外侧悬空检查: 棱的外面应该是空的 (深度突然变远 / 无效), 夹爪才伸得进去."""
    h_img, w_img = depth_mm.shape[:2]
    d = Q - P
    L = np.linalg.norm(d)
    if L < 1e-6:
        return 0.0
    d = d / L
    free = tot = 0
    for t in np.linspace(0.08, 0.92, 20):
        for off in (config.BCV_FREE_PX, config.BCV_FREE_PX * 2):
            base = P + t * L * d + out_n * off
            u, v = int(round(base[0])), int(round(base[1]))
            if not (0 <= u < w_img and 0 <= v < h_img):
                continue
            tot += 1
            if mask_blob[v, u]:
                continue                      # 还在物体上, 不算空
            z = depth_mm[v, u]
            if z == 0 or z >= config.MAX_DEPTH_MM:
                free += 1                     # 无效深度 = 通常是空
            elif z / 1000.0 - z_edge > config.BCV_FREE_DZ:
                free += 1                     # 明显更远 = 空
    return float(free) / max(tot, 1)


def _size_score(L):
    """长度对上实物尺寸的程度. 文件筐 308 x 303 x 257mm.

    保底分给得很吝啬 (平方衰减): 0.12m 的碎线对夹爪根本没用, 不能让它凭
    "位置高、点少所以拟合得漂亮"就赢过一条完整的 0.30m 顶沿.
    """
    g = max(_gauss(L, 0.308, 0.060),
            _gauss(L, 0.303, 0.060),
            _gauss(L, 0.257, 0.060))
    partial = 0.35 * float(np.clip(L / 0.30, 0.0, 1.0)) ** 2
    return float(np.clip(max(g, partial), 0.0, 1.0))


def best_grasp_line(mask_blob, depth_mm, intr, plane, blob3d, dbg=None):
    """安全禁用旧单最高棱接口，避免外部调用方绕过五板红线语义。"""
    del mask_blob, depth_mm, intr, plane, blob3d
    if dbg is not None:
        dbg["reject"] = "legacy_single_highest_edge_disabled"
    return None


def _legacy_best_grasp_line_reference(
    mask_blob, depth_mm, intr, plane, blob3d, dbg=None
):
    """在一个蓝色 blob 上找**最佳**抓取线 (只返回置信度最高的一条)"""
    contour, segs = _contour_segments(mask_blob)
    up, _, _ = _ground_frame(plane)
    h_top = float(np.percentile(blob3d["h"], 98))
    # 高度一律换算成"在这个物体自己身上的相对高度", 而不是"离地多高".
    # 为什么: 筐放在**桌子上**时, 它整个身子离地都有 0.75m, 于是"离地 0.45*h_top"
    # 这个条件连筐的**底边**都能满足 —— 实测 124754_039 就把线画在了筐和桌面的
    # 交线上 (夹爪根本伸不进去). 换成物体自身的相对高度以后, 桌上桌下一个样.
    h_base = float(np.percentile(blob3d["h"], 2))
    h_raw = h_top - h_base                     # 这块 blob 自己量出来的高度跨度
    h_span = max(h_raw, 0.06)
    z_ref0 = float(blob3d["z_near"])
    if dbg is not None:
        dbg["h_base"] = round(h_base, 3)
        dbg["h_span"] = round(h_raw, 3)

    # === 候选边的两个来源 ===
    #  A. 蓝掩膜的外轮廓 (侧影)          —— 筐侧躺/斜放时顶沿就在侧影上
    #  B. **最高那一层面的轮廓**          —— 筐倒扣或从上往下看时, 真正该抓的棱
    #                                       在侧影里面, 只有按高度切一层才找得到
    sources = []
    if contour is not None and segs:
        sources.append((contour, segs, mask_blob, "sil"))
    try:
        hmap, okmap, zmap = _height_image(depth_mm, intr, plane)
        band = _top_band_mask(mask_blob, hmap, okmap, zmap, z_ref0, h_top)
    except Exception:
        band = None
    if band is not None:
        n_lab, lab_b, st_b, _ = cv2.connectedComponentsWithStats(band, 8)
        idx = sorted(range(1, n_lab),
                     key=lambda i: -st_b[i, cv2.CC_STAT_AREA])[:3]
        for i in idx:
            if st_b[i, cv2.CC_STAT_AREA] < config.BCV_TOP_MIN_AREA:
                continue
            sub = (lab_b == i).astype(np.uint8)
            cs, _h = cv2.findContours(sub, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
            if not cs:
                continue
            cc = max(cs, key=cv2.contourArea)
            ct, sg = _one_contour_segments(cc)
            if ct is not None and sg:
                sources.append((ct, sg, sub, "top"))
        # C. **跨连通域的 RANSAC 共线拟合** —— 顶沿被打断成好几截时的救命稻草.
        #    支撑点集直接用整块顶面带的像素, 于是 cover/maxgap 量的正好是
        #    "这条线全长有没有顶面像素托着", 正是我们要的.
        bpts = np.stack(np.nonzero(band)[::-1], axis=1).astype(np.float64)
        for (P0, Q0) in _band_ransac(band):
            sources.append((bpts, [(P0, Q0)], band, "rans"))
    if not sources:
        return None
    # "上"在图像里的方向: 取物体中心和它上方 10cm 的投影之差
    ctr3 = blob3d["pts"].mean(axis=0)
    uv0 = _project(ctr3, intr)
    uv1 = _project(ctr3 + 0.10 * up, intr)
    up2 = uv1 - uv0
    up2 = up2 / max(np.linalg.norm(up2), 1e-9)

    z_ref = float(blob3d["z_near"])
    cands = []
    rej = {"short": 0, "bent": 0, "air": 0, "down": 0, "nodepth": 0,
           "sparse": 0, "fit": 0, "len": 0}
    for (contour, segs, src_mask, src_tag) in sources:
      for (P0, Q0) in segs:
        P, Q, dev2d, npt, cover, maxgap = _refit_2d(contour, P0, Q0)
        if np.linalg.norm(Q - P) < config.BCV_MIN_SEG_PX:
            rej["short"] += 1
            continue
        if dev2d > config.BCV_MAX_DEV_PX:
            rej["bent"] += 1
            continue                                  # 2D 上就不直, 不是棱
        if maxgap > config.BCV_MAX_GAP or cover < config.BCV_MIN_COVER:
            rej["air"] += 1
            continue                                  # 中间悬空, 是凸包的假弦
        # 外法向要相对**这条边自己所属的那块面**来判, 不是相对整个筐
        out_n = _outward_normal(src_mask, P, Q)
        if src_tag == "rans":
            # RANSAC 那条线是穿在顶面带**中间**的, 两侧都有掩膜像素, 数像素
            # 分不出内外. 改成直接问深度: 哪一侧更空, 夹爪就从哪一侧进.
            zg = float(np.median(depth_mm[
                np.clip(np.round(np.linspace(P[1], Q[1], 12)).astype(int),
                        0, depth_mm.shape[0] - 1),
                np.clip(np.round(np.linspace(P[0], Q[0], 12)).astype(int),
                        0, depth_mm.shape[1] - 1)])) / 1000.0
            if zg > 0.1:
                fa = _free_side(P, Q, out_n, depth_mm, mask_blob, zg)
                fb = _free_side(P, Q, -out_n, depth_mm, mask_blob, zg)
                if fb > fa:
                    out_n = -out_n
        # 只毙掉外法向明显朝下的 (= 物体和地面的交线). 真正判"能不能抓"的是
        # 离地高度 s_h, 不是它在图像里朝哪边.
        upness = float(out_n @ up2)
        if upness < config.BCV_MIN_UPNESS:
            rej["down"] += 1
            continue
        lifted = _lift_segment(P, Q, out_n, depth_mm, intr, plane,
                               mask_blob, z_ref)
        if lifted is None:
            rej["nodepth"] += 1
            continue
        if lifted["valid_frac"] < config.BCV_MIN_VALID_FRAC:
            rej["sparse"] += 1
            continue
        ctr, dvec, rms, keep = _fit_line3d(lifted["pts"])
        if keep.sum() < 8:
            rej["fit"] += 1
            continue
        kp = lifted["pts"][keep]
        # === 端点: 由 **2D 轮廓端点** 的视线与 3D 直线求交得到 ===
        # 而不是取深度采样点的最远两端 —— 后者只要有一段没深度就会缩水.
        A3 = _ray_line_point(P, intr, ctr, dvec)
        B3 = _ray_line_point(Q, intr, ctr, dvec)
        if A3 is None or B3 is None:
            t = (kp - ctr) @ dvec
            A3 = ctr + t.min() * dvec
            B3 = ctr + t.max() * dvec
            Auv, Buv = _project(A3, intr), _project(B3, intr)
        else:
            Auv, Buv = P.copy(), Q.copy()      # 画图直接用轮廓端点 -> 严丝合缝
        Lm = float(np.linalg.norm(B3 - A3))
        if Lm > config.BCV_MAX_LEN or Lm < config.BCV_MIN_LEN:
            rej["len"] += 1
            continue
        hA = float(_height(A3[None, :], plane)[0])
        hB = float(_height(B3[None, :], plane)[0])
        h_mid = 0.5 * (hA + hB)
        z_edge = float(np.median(kp[:, 2]))
        freef = _free_side(P, Q, out_n, depth_mm, mask_blob, z_edge)

        h_rel = (h_mid - h_base) / h_span        # 0 = 物体底边, 1 = 物体顶边
        s_h = float(np.clip(h_rel, 0.0, 1.0)) ** 1.5
        s_sz = _size_score(Lm)
        s_rms = _gauss(rms, 0.0, config.BCV_RMS_SIGMA)
        s_2d = float(np.clip(1.0 - dev2d / config.BCV_MAX_DEV_PX, 0.0, 1.0))
        s_free = float(np.clip(freef / 0.7, 0.0, 1.0))
        # 连续性分: 主要看"最大空档"(越连续越好), 覆盖率只占小头 —— 锯齿顶沿的
        # 覆盖率本来就低, 不能因此判它低分.
        s_cov = float(np.clip(
            0.65 * (1.0 - maxgap / max(config.BCV_MAX_GAP, 1e-6)) +
            0.35 * np.clip((cover - config.BCV_MIN_COVER) /
                           max(0.75 - config.BCV_MIN_COVER, 1e-6), 0.0, 1.0),
            0.0, 1.0))
        # 长度(s_sz)的权重调到最高: 一条完整的顶沿才是能用的抓取线,
        # 碎片就算位置高、拟合漂亮也不该赢.
        conf = (0.24 * s_h + 0.30 * s_sz + 0.12 * s_rms +
                0.08 * s_2d + 0.16 * s_free + 0.10 * s_cov)
        # 水平度: 抓取棱基本是水平的 (筐怎么放, 顶上那条棱都近似水平)
        horiz = 1.0 - abs(float(dvec @ up))
        conf *= float(np.clip(0.55 + 0.45 * horiz / max(1e-6, 1.0), 0.0, 1.0))
        cands.append({
            "conf": float(np.clip(conf, 0, 1)),
            "A3": A3, "B3": B3, "Auv": Auv, "Buv": Buv,
            "len": Lm, "rms": rms, "dev2d": dev2d, "h_mid": h_mid,
            "free": freef, "upness": upness, "n": int(keep.sum()),
            "cover": cover, "maxgap": maxgap, "src": src_tag,
            "s": (s_h, s_sz, s_rms, s_2d, s_free, s_cov),
            "h_rel": round(h_rel, 3),
            "top": h_rel >= config.BCV_TOP_H_RATIO,
        })
    n_seg_all = sum(len(s[1]) for s in sources)
    if not cands:
        if dbg is not None:
            dbg["n_cand"] = 0
            dbg["rej"] = rej
            dbg["n_seg"] = n_seg_all
        return None
    # 优先只在"确实靠近物体顶部"的候选里选 —— 贴地那条是物体和地面的交线,
    # 夹爪根本伸不进去. 但如果一条都没有, 就退回全体, 保证不空手 (宁可给一条
    # 次优的线, 也不能一条都不给).
    tops = [c for c in cands if c["top"]]
    pool = tops if tops else cands
    pool.sort(key=lambda c: -c["conf"])
    if dbg is not None:
        dbg["n_cand"] = len(cands)
        dbg["n_top"] = len(tops)
        dbg["n_seg"] = n_seg_all
        dbg["n_src"] = len(sources)
        dbg["rej"] = rej
        dbg["top3"] = [(round(c["conf"], 3), round(c["len"], 3),
                        round(c["h_mid"], 3), c["src"]) for c in pool[:3]]
        # 全部候选的明细, 只给离线诊断脚本用 (画图 + 查为什么某条线赢了).
        dbg["cands"] = [{"conf": round(c["conf"], 3), "len": round(c["len"], 3),
                         "src": c["src"], "top": bool(c["top"]),
                         "h_rel": c["h_rel"], "free": round(c["free"], 2),
                         "cover": round(c["cover"], 2),
                         "maxgap": round(c["maxgap"], 2),
                         "rms": round(c["rms"] * 1000, 1), "n": c["n"],
                         "s": [round(x, 3) for x in c["s"]],
                         "uv": [[float(c["Auv"][0]), float(c["Auv"][1])],
                                [float(c["Buv"][0]), float(c["Buv"][1])]]}
                        for c in sorted(cands, key=lambda c: -c["conf"])]
    b = pool[0]
    return {
        "type": "line",
        "name": "basket_grasp_edge",
        "endpoints": [b["A3"].tolist(), b["B3"].tolist()],
        "endpoints_uv": [[float(b["Auv"][0]), float(b["Auv"][1])],
                          [float(b["Buv"][0]), float(b["Buv"][1])]],
        "length_m": float(b["len"]),
        "confidence": float(b["conf"]),
        "orientation": "up",
        "height_above_ground": float(b["h_mid"]),
        "rms_mm": float(b["rms"] * 1000),
        "debug": {"dev2d_px": round(b["dev2d"], 2), "free": round(b["free"], 2),
                   "upness": round(b["upness"], 2), "n_pts": b["n"],
                   "cover": round(b["cover"], 2),
                   "maxgap": round(b["maxgap"], 2), "src": b["src"],
                   "scores": [round(x, 3) for x in b["s"]]},
    }


# ============================================================
# 5. 主入口
# ============================================================

def detect_baskets(
    rgb_bgr,
    depth_mm,
    intr,
    plane,
    max_n=3,
    *,
    include_grasps=True,
):
    """颜色优先的文件筐检测. 返回候选列表 (已按置信度降序)."""
    if rgb_bgr is None:
        return []
    dbgm = {}
    mask, lab, stats, keep, (H, S, V, BmR) = blue_mask(rgb_bgr, dbgm)
    if mask.sum() == 0:
        return []
    img_h, img_w = depth_mm.shape[:2]
    out = []
    order = [i for i in range(1, len(stats)) if keep[i]]
    order.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    blue_raw = mask
    for i in order[:max_n + 2]:
        x, y, w, h, area = stats[i]
        # 机器人自己的躯干: 永远贴着画面底边的一条窄带 (头部相机和身体刚性连接)
        if (y + h) >= img_h - 3 and y >= config.CV_ROBOT_BODY_TOP_RATIO * img_h:
            continue
        blob = (lab == i).astype(np.uint8)
        b3 = _blob_points(blob, depth_mm, intr, plane)
        if b3 is None:
            continue
        pts, hh = b3["pts"], b3["h"]
        ctr3 = pts.mean(axis=0)
        dist = float(np.linalg.norm(ctr3[[0, 2]]))
        if dist > config.BCV_MAX_DIST or dist < config.BCV_MIN_DIST:
            continue
        h_top = float(np.percentile(hh, 98))

        # ---- 尺寸: 地面内的最小外接矩形边长 ----
        up, e1, e2 = _ground_frame(plane)
        ab = np.stack([pts @ e1, pts @ e2], axis=1)
        fp_long, fp_short = _obb_2d(ab)

        # ---- 是否被画面边缘裁切 ----
        # 裁切了的话, 量到的占地长边只是**下界**, 不能拿它去否决.
        # 实测: 3 个占地量不到 0.30 的真筐, 全都是被边缘裁掉的; 9 个垃圾桶
        # 候选一个都没贴边. 这一条把"尺寸偏小"的两种原因区分开了.
        border = bool(x <= 2 or y <= 2 or (x + w) >= img_w - 3 or (y + h) >= img_h - 3)

        # ---- 证据一: 色度 (与光照/距离无关) ----
        # rel = (B-R)/max(B,G,R) = 1 - R/B. 绝对差 B-R 只用来分割像素(见 blue_mask),
        # **不参与打分** —— 它随距离和光照整体缩放, 同一个灰蓝垃圾桶近拍就能骗到满分.
        bm = blob.astype(bool)
        bmr_all = BmR[bm]
        bmr_med = float(np.median(bmr_all))
        bmr_p75 = float(np.percentile(bmr_all, 75))
        s_med = float(np.median(S[bm]))
        v_med = float(np.median(V[bm]))
        h_med = float(np.median(H[bm]))
        rel = bmr_all.astype(np.float64) / np.maximum(V[bm].astype(np.float64), 1.0)
        rel_med = float(np.median(rel))
        rel_p70 = float(np.percentile(rel, 70))
        rel_p75 = float(np.percentile(rel, 75))
        # 取 70/75 两个分位的平均: 中位数会被"筐一半在阴影里"压下去(和桶重叠),
        # 而更高的分位(85+)又开始采到高光噪点. 平均相邻两个分位是为了不依赖
        # "刚好某一个分位好用"这种巧合.
        rel_hi = 0.5 * (rel_p70 + rel_p75)
        s_chr = max(_ramp(rel_hi, config.BCV_RELHI_REJECT, config.BCV_RELHI_FULL),
                    _ramp(rel_med, config.BCV_REL_REJECT, config.BCV_REL_FULL))

        # ---- 证据二: 占地长边 (筐 0.308m 见方; 垃圾桶直径约 0.25m) ----
        s_fp = _ramp(fp_long, config.BCV_FP_REJECT, config.BCV_FP_FULL)
        if border:
            s_fp = max(s_fp, config.BCV_FP_BORDER)

        # ---- 软与: 两条证据都得说得过去 ----
        f = config.BCV_GATE_FLOOR
        gate = (((f + (1 - f) * s_chr) ** config.BCV_GATE_W) *
                ((f + (1 - f) * s_fp) ** (1 - config.BCV_GATE_W)))

        # ---- 尺寸细分 / 结构: 只做微调, 不做否决 ----
        # fp_long 只对 0.308 一个峰: 257mm 永远是三边里最短的那条, 不可能当长边.
        s_size = (0.45 * _ngauss(fp_long, 0.305, 0.085) +
                  0.25 * max(_ngauss(fp_short, 0.303, 0.10), _ngauss(fp_short, 0.257, 0.10)) +
                  0.30 * max(_ngauss(h_top, 0.257, 0.10), _ngauss(h_top, 0.303, 0.10)))
        if border:
            s_size = max(s_size, config.BCV_FP_BORDER)
        s_size = float(np.clip(s_size, 0, 1))

        # ---- 结构分数 (五棱 + 网洞) ----
        s_ridge, n_ridge, gap = _ridge_score(pts, hh, plane)
        s_hole, hole_ratio = _hole_score(blob, blue_raw)
        s_struct = float(np.clip(0.6 * s_ridge + 0.4 * s_hole, 0, 1))

        conf = gate * (config.BCV_CONF_BASE +
                       config.BCV_CONF_K_SIZE * s_size +
                       config.BCV_CONF_K_STRUCT * s_struct)
        # 两条证据都很强 + 像素够多 -> 给个保底分, 绝不漏检.
        if gate >= config.BCV_SURE_GATE and area >= config.BCV_SURE_AREA_PX:
            conf = max(conf, config.BCV_SURE_CONF)
        conf = float(np.clip(conf, 0, 1))
        s_color = s_chr                      # 兼容旧的调试脚本

        # 置信度不是只用于展示：低分候选必须在进入抓取语义和标准对象接口前
        # 被拒绝。灰蓝垃圾桶在回归集中会形成蓝色候选，但最高仅约 0.26；
        # 真文件筐训练集最低为 0.592。
        if conf < config.BCV_DETECT_CONF_MIN:
            continue

        gdbg = {}
        grasps = (
            extract_basket_grasps(
                rgb_bgr,
                blob,
                depth_mm,
                intr,
                plane,
                b3,
                gdbg,
            )
            if include_grasps
            else []
        )
        out.append({
            "cls": "basket", "confidence": conf,
            "center_3d": ctr3, "points": pts,
            "n_points": int(len(pts)),
            "height_above_ground": h_top,
            "fp_long": fp_long, "fp_short": fp_short,
            "footprint_area": float(fp_long * fp_short),
            "pixel_bbox": (int(x), int(y), int(w), int(h)),
            "mask": blob,
            # ``grasp`` 仅留作旧调用方兼容；真实结果严格受红线语义约束，
            # ``grasps`` 只能为 [] 或 [唯一最高置信安全线]。
            "grasps": grasps,
            "grasp": grasps[0] if grasps else None,
            "debug": {
                "area_px": int(area), "bmr": round(bmr_med, 1),
                "bmr75": round(bmr_p75, 1),
                "S": round(s_med, 1), "V": round(v_med, 1), "Hh": round(h_med, 1),
                "rel": round(rel_med, 3), "rel75": round(rel_p75, 3),
                "rel_hi": round(rel_hi, 3), "border": int(border),
                "s_chr": round(s_chr, 3), "s_fp": round(s_fp, 3),
                "gate": round(gate, 3),
                "s_color": round(s_color, 3), "s_size": round(s_size, 3),
                "s_struct": round(s_struct, 3),
                "s_ridge": round(s_ridge, 3), "n_ridge": n_ridge,
                "ridge_gap": round(gap, 3),
                "s_hole": round(s_hole, 3), "hole_ratio": round(hole_ratio, 3),
                "dist": round(dist, 2),
                "grasp_algorithm": (
                    GRASP_ALGORITHM_VERSION if include_grasps else None
                ),
                "grasp": gdbg,
            },
        })
    out.sort(key=lambda o: -o["confidence"])
    return out[:max_n]
