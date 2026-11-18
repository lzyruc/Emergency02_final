"""可视化 Phase 1 结果"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .geometry import project_to_image


CLASS_COLORS_BGR = {
    "box": (60, 130, 200),        # 棕橙
    "chair": (180, 60, 200),      # 紫
    "trash_can": (60, 200, 200),  # 青
    "basket": (200, 180, 60),     # 蓝
    "unknown": (128, 128, 128),
}


def _to_bgr(c):
    return (int(c[0]), int(c[1]), int(c[2]))


def draw_annotations_on_rgb(rgb_bgr, result, intr):
    """在 RGB 图上叠加抓取标注"""
    img = rgb_bgr.copy()
    for obj in result.objects:
        color = _to_bgr(CLASS_COLORS_BGR.get(obj.cls, (200, 200, 200)))
        # 专用检测器可在 debug 中附带 xywh 像素框。画框能直观看出识别范围，
        # 同时不改变原有 ObjectResult / JSON 接口。
        bbox = obj.debug.get("pixel_bbox") if obj.debug else None
        if (
            bbox
            and len(bbox) == 4
            and obj.debug.get("draw_bbox", True)
        ):
            x, y, w, h = (int(round(v)) for v in bbox)
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 2)

        # 文件筐抓取线和垃圾桶口沿需要逐像素目检；中心圆和长标签会挡住
        # 真实边缘，所以把类别标签挪到框角。其他类别保持中心标注方式。
        center_uv = project_to_image(np.array([obj.position_cam]), intr)[0]
        if obj.cls in {"basket", "trash_can"} and bbox and len(bbox) == 4:
            label_y = max(18, int(round(bbox[1])) - 7)
            cv2.putText(
                img,
                f"{obj.id}:{obj.cls} ({obj.confidence:.1f})",
                (int(round(bbox[0])), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        else:
            cv2.circle(img, (int(center_uv[0]), int(center_uv[1])), 6, color, 2)
            cv2.putText(img, f"{obj.id}:{obj.cls} ({obj.confidence:.1f})",
                        (int(center_uv[0]) + 10, int(center_uv[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for a in obj.grasp_annotations:
            if a["type"] == "line":
                # v14: 如果标注自带 2D 像素端点, 直接用它.
                # 理由: 这条线本来就是在图像上沿物体轮廓定出来的, 直接画 = 严丝合缝;
                # 而"拟合成 3D 再投影回来"会把深度噪声放大成几个到十几个像素的偏移,
                # 看上去线就飘在物体旁边. 3D 端点仍然保留在 endpoints 里给机械臂用.
                if a.get("endpoints_uv"):
                    p1 = np.asarray(a["endpoints_uv"][0], dtype=float)
                    p2 = np.asarray(a["endpoints_uv"][1], dtype=float)
                else:
                    p1 = project_to_image(np.array([a["endpoints"][0]]), intr)[0]
                    p2 = project_to_image(np.array([a["endpoints"][1]]), intr)[0]
                line_color = (
                    (0, 0, 255)
                    if a.get("name") == "basket_divider_free_edge"
                    else color
                )
                cv2.line(img,
                          (int(p1[0]), int(p1[1])),
                          (int(p2[0]), int(p2[1])),
                          line_color,
                          2 if a.get("name") == "basket_divider_free_edge" else 3)
                # orientation 标签
                mid = ((p1 + p2) / 2).astype(int)
                if a.get("name") == "basket_divider_free_edge":
                    # 文件筐只画唯一 Top-1 红线且不压文字，便于逐像素核对。
                    line_label = ""
                else:
                    line_label = (
                        "top"
                        if a.get("name") == "backrest_top_edge"
                        else a.get("orientation", "")
                    )
                if line_label:
                    cv2.putText(img, line_label,
                                (mid[0], mid[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
            elif a["type"] == "circle":
                # 真实圆口在透视图中是椭圆，绝不能再用一个像素半径画正圆。
                # 专用垃圾桶分支保存的是直接贴合桶沿的 OpenCV RotatedRect。
                ellipse_uv = a.get("ellipse_uv")
                circle_color = (
                    (0, 0, 255)
                    if a.get("name") == "top_rim"
                    else color
                )
                if isinstance(ellipse_uv, dict):
                    ellipse_center = ellipse_uv.get("center")
                    ellipse_size = ellipse_uv.get("size")
                    if (
                        ellipse_center
                        and len(ellipse_center) == 2
                        and ellipse_size
                        and len(ellipse_size) == 2
                    ):
                        cv2.ellipse(
                            img,
                            (
                                (
                                    float(ellipse_center[0]),
                                    float(ellipse_center[1]),
                                ),
                                (
                                    float(ellipse_size[0]),
                                    float(ellipse_size[1]),
                                ),
                                float(ellipse_uv.get("angle_deg", 0.0)),
                            ),
                            circle_color,
                            2,
                            cv2.LINE_AA,
                        )
                        continue

                contour_uv = a.get("contour_uv")
                if contour_uv:
                    contour = np.rint(
                        np.asarray(contour_uv, dtype=float)
                    ).astype(np.int32)
                    if contour.ndim == 2 and contour.shape[0] >= 3:
                        cv2.polylines(
                            img,
                            [contour.reshape(-1, 1, 2)],
                            True,
                            circle_color,
                            2,
                            cv2.LINE_AA,
                        )
                        continue

                # 兼容旧 JSON：从 3D 法向量构造圆面正交基，采样后逐点投影。
                center = np.asarray(a["center"], dtype=float)
                radius = float(a["radius"])
                normal = np.asarray(
                    a.get("normal", [0.0, 1.0, 0.0]),
                    dtype=float,
                )
                normal /= np.linalg.norm(normal) + 1e-12
                axis_u = np.asarray(a.get("axis_u", []), dtype=float)
                axis_v = np.asarray(a.get("axis_v", []), dtype=float)
                if axis_u.shape != (3,) or axis_v.shape != (3,):
                    reference = np.array([0.0, 0.0, 1.0])
                    axis_u = np.cross(normal, reference)
                    if np.linalg.norm(axis_u) < 1e-6:
                        reference = np.array([1.0, 0.0, 0.0])
                        axis_u = np.cross(normal, reference)
                    axis_u /= np.linalg.norm(axis_u) + 1e-12
                    axis_v = np.cross(normal, axis_u)
                    axis_v /= np.linalg.norm(axis_v) + 1e-12
                theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
                circle_3d = (
                    center[None, :]
                    + radius * np.cos(theta)[:, None] * axis_u[None, :]
                    + radius * np.sin(theta)[:, None] * axis_v[None, :]
                )
                projected = np.rint(
                    project_to_image(circle_3d, intr)
                ).astype(np.int32)
                cv2.polylines(
                    img,
                    [projected.reshape(-1, 1, 2)],
                    True,
                    circle_color,
                    2,
                    cv2.LINE_AA,
                )
            elif a["type"] == "point":
                # v11.9: 分隔格中心点 → 画一个双圈 + 十字
                pos = np.array(a["position"])
                uv_p = project_to_image(np.array([pos]), intr)[0]
                px, py = int(uv_p[0]), int(uv_p[1])
                cv2.circle(img, (px, py), 10, color, 2)
                cv2.circle(img, (px, py), 3, color, -1)
                cv2.line(img, (px - 14, py), (px + 14, py), color, 1)
                cv2.line(img, (px, py - 14), (px, py + 14), color, 1)
                # 标 slot index
                idx = a.get("name", "").split("_")[-1]
                cv2.putText(img, f"slot{idx}", (px + 12, py - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if obj.cls == "chair" and not obj.grasp_annotations:
            bx, by = (int(center_uv[0]) + 10, int(center_uv[1]) + 24)
            cv2.putText(img, "top edge unavailable: no safe grasp line",
                        (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 80, 255), 1)
    return img


def visualize_result(depth_mm, rgb_bgr, result, intr, save_path=None, title=""):
    """5 宫格可视化 (RGB + 深度 + 俯视 + 每类抓取 + stats 文本)"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # 深度伪彩
    depth_vis = depth_mm.astype(np.float32)
    depth_vis[depth_mm == 0] = np.nan
    axes[0, 0].imshow(depth_vis, cmap='jet', vmin=200, vmax=5000)
    axes[0, 0].set_title("Depth (mm)")
    axes[0, 0].axis('off')

    # RGB (可能没有, 就画深度灰度)
    if rgb_bgr is not None:
        axes[0, 1].imshow(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 1].imshow(depth_vis, cmap='gray')
    axes[0, 1].set_title("RGB reference")
    axes[0, 1].axis('off')

    # RGB + 抓取标注叠加
    if rgb_bgr is not None:
        annotated = draw_annotations_on_rgb(rgb_bgr, result, intr)
        axes[0, 2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    else:
        depth_bgr = cv2.applyColorMap(
            np.clip(depth_mm.astype(np.float32) / 20, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET)
        annotated = draw_annotations_on_rgb(depth_bgr, result, intr)
        axes[0, 2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Annotations")
    axes[0, 2].axis('off')

    # 俯视点云: 展示每个 kept object
    for obj in result.objects:
        color = np.array(CLASS_COLORS_BGR.get(obj.cls, (128, 128, 128))) / 255.0
        # 用 obj.principal_extents + axes 画一个投影框
        u1 = np.array(obj.principal_axes[0])
        u2 = np.array(obj.principal_axes[1])
        d1 = obj.principal_extents[0]
        d2 = obj.principal_extents[1]
        center = np.array(obj.position_cam)
        # 4 角
        corners = []
        for s1 in [-0.5, 0.5]:
            for s2 in [-0.5, 0.5]:
                corners.append(center + s1 * d1 * u1 + s2 * d2 * u2)
        corners = np.array(corners)
        # 按顺时针排 (X-Z)
        cx, cz = corners[:, 0].mean(), corners[:, 2].mean()
        angles = np.arctan2(corners[:, 2] - cz, corners[:, 0] - cx)
        order = np.argsort(angles)
        poly_xz = corners[order][:, [0, 2]]
        axes[1, 0].fill(poly_xz[:, 0], poly_xz[:, 1], color=color, alpha=0.4)
        axes[1, 0].plot(np.append(poly_xz[:, 0], poly_xz[0, 0]),
                        np.append(poly_xz[:, 1], poly_xz[0, 1]),
                        color=color, linewidth=1.5)
        axes[1, 0].text(obj.position_cam[0], obj.position_cam[2],
                        f"{obj.id}:{obj.cls}", fontsize=9,
                        ha='center', bbox=dict(boxstyle='round',
                                                facecolor='white', edgecolor=color, alpha=0.7))
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Z (m, forward)')
    axes[1, 0].set_title(f'Top-down (kept={result.n_kept})')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(-3, 3)
    axes[1, 0].set_ylim(4, 0)

    # 统计文本
    axes[1, 1].axis('off')
    lines = [f"ground plane: [{result.ground_plane[0]:.3f}, "
             f"{result.ground_plane[1]:.3f}, {result.ground_plane[2]:.3f}, "
             f"{result.ground_plane[3]:.3f}]",
             f"raw clusters: {result.n_raw_clusters}, kept: {result.n_kept}",
             "",
             f"{'ID':>2} {'class':>10} {'cnf':>4} {'H_cm':>5} "
             f"{'d1':>5} {'d2':>5} {'d3':>5} {'solid':>5} {'grasps':>7}"]
    for obj in result.objects:
        d = obj.principal_extents
        lines.append(f"{obj.id:>2} {obj.cls:>10} {obj.confidence:>4.2f} "
                     f"{obj.height_above_ground * 100:>5.1f} "
                     f"{d[0]:>5.2f} {d[1]:>5.2f} {d[2]:>5.2f} "
                     f"{obj.solid_ratio:>5.2f} {len(obj.grasp_annotations):>7d}")
    lines.append("")
    lines.append("orientations of grasps:")
    for obj in result.objects:
        os = [a.get("orientation", "?") for a in obj.grasp_annotations]
        if os:
            lines.append(f"  {obj.id} {obj.cls}: {os}")
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va='top', ha='left',
                    fontfamily='monospace', fontsize=8,
                    transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('stats')

    # 每个物体的 debug info
    axes[1, 2].axis('off')
    lines = ["classifier reasons:"]
    for obj in result.objects:
        lines.append(f"{obj.id} {obj.cls}: {obj.debug.get('reason', '')}")
    axes[1, 2].text(0.02, 0.98, "\n".join(lines), va='top', ha='left',
                    fontfamily='monospace', fontsize=8,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('classifier debug')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
