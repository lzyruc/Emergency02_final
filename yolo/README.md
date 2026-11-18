# YOLO 识别模块

## 应急四类障碍物

应急 Ultralytics 权重：

```text
models/emerency_yolo26l_20260804_cls1234_best.pt
```

模型类别映射：

```text
1 -> chair
2 -> filebox
3 -> bin
4 -> box
```

调用：

```python
from yolo import ObstacleYolo

detector = ObstacleYolo()
detections = detector.detect(bgr_image)
```

输出为二维检测结果：

```python
[
    {
        "type": "chair",
        "confidence": 0.91,
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "center_uv": [u, v],
    }
]
```

`bgr_image` 为相机原始 `uint8 H×W×3` BGR 图像。模型实例应在任务启动时创建一次，
连续帧复用，不要在识别热路径重复加载权重。运行时依赖 Ultralytics、
PyTorch 和 NumPy，默认使用 CUDA:0 的 FP16 推理。

## 原商超商品识别

输入 `640x480` RGB 图、对齐后的深度图和商品 ID/名称，返回商品中心像素坐标和平均深度。

## 文件

```text
product_yolo.py          识别接口
test.py                  RealSense 实时测试
商品ID名称表.json         ID 与名称映射
models/                  YOLO 模型
```

## 调用

```python
from product_yolo import detect_products

results = detect_products(
    rgb_image,
    depth_image,
    ["45", "康师傅香辣牛肉面", 66],
)
```

输入要求：

- `rgb_image`：形状为 `(480, 640, 3)` 的 RGB 图。
- `depth_image`：形状为 `(480, 640)`，且已与 RGB 图对齐。
- 第三个参数：商品 ID 或名称列表，支持混合输入。

返回示例：

```python
{
    "45": [
        {
            "center": (312, 267),
            "average_depth": 0.428
        }
    ],
    "51": [],
    "66": []
}
```

- 返回键统一为字符串 ID。
- `center` 格式为 `(x, y)`，左上角是 `(0, 0)`。
- 深度单位与输入深度图相同。
- 未识别到时返回空列表。
- 同类商品出现多个时，只返回深度最小的最近实例。

## 实时测试

```powershell
cd "C:\Users\Lenovo\Desktop\机器人比赛\识别模块"
& "C:\Users\Lenovo\Desktop\机器人比赛\vision-yolo-detr\.venv\Scripts\python.exe" .\test.py
```

启动后输入一个或多个商品 ID：

```text
45
45 46 47
45,46,47
```

快捷键：

- `Q` 或 `Esc`：退出。
- `S`：保存当前预览图。

指定相机或直接指定商品：

```powershell
python .\test.py --serial 260622272412
python .\test.py --products 45 46 47
```

`test.py` 会对齐 RGB 和深度图，并将深度换算为米。

## 模型

| 商品 ID | 模型文件 |
|---|---|
| 1-35 | `models/shelf1_yolo26l_best.pt` |
| 36-66 | `models/shelf2_yolo26l_best.pt` |
| 67-89 | `models/shelf3_yolo26l_best.pt` |
| 90-107 | `models/shelf4_yolo26l_best.pt` |

替换模型时，使用对应文件名覆盖原 `.pt` 文件，然后重启程序。

模型类别名称必须是商品 ID，例如 `"36"`、`"45"`。缺少某个模型时，对应商品返回空列表。

目前只支持商超场景，暂时只有第二个货架的模型。后续可以拓展为应记和酒店场景。

## 参数

参数位于 `product_yolo.py` 顶部：

```python
CONFIDENCE_THRESHOLD = 0.25
DEPTH_RADIUS_RATIO = 0.30
```

- `CONFIDENCE_THRESHOLD`：置信度必须严格大于该值。
- `DEPTH_RADIUS_RATIO`：深度采样圆半径占检测框宽度的比例。

深度计算会过滤 `0`、负数、`NaN` 和无穷值，再计算采样区域平均值。

## 依赖

```text
numpy
torch
ultralytics
```

运行 `test.py` 还需要：

```text
opencv-python
pyrealsense2
```
