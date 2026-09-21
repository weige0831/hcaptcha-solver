"""
通用视觉检测层
图像质量评估 / 智能预处理 / 类别映射 / 检测模型管理

被 hcaptcha_solver.py 复用，不包含任何特定验证码平台的逻辑。
"""

from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import numpy as np
import cv2
import io
import os
import torch
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 配置 ---
DEBUG = os.environ.get("SOLVER_DEBUG", "1") == "1"
ENABLE_IMAGE_PREPROCESSING = os.environ.get("SOLVER_PREPROCESS", "1") == "1"
GROUNDING_DINO_CONFIDENCE = 0.25
GROUNDING_DINO_DEBUG = False

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11x.pt")
GROUNDING_MODEL_ID = os.environ.get("GROUNDING_MODEL_ID", "IDEA-Research/grounding-dino-tiny")

SCREENSHOT_DIR = os.environ.get("SOLVER_SCREENSHOT_DIR", "debug_screenshots")

# GroundingDINO 提示词映射（零样本类别 -> 英文提示词）
GROUNDING_PROMPTS = {
    "crosswalk": "zebra crossing. crosswalk.",
    "人行横道": "zebra crossing. pedestrian crossing. road stripes.",
    "stair": "staircase. stairs. steps.",
    "楼梯": "staircase. stairs. steps.",
    "bridge": "bridge. overpass.",
    "桥": "bridge. overpass.",
    "chimney": "chimney. smokestack.",
    "烟囱": "chimney. smokestack.",
}

# 题面关键词 -> 检测类别
CATEGORY_MAPPING = {
    "摩托": ["motorcycle"], "motorcycle": ["motorcycle"],
    "公交": ["bus"], "巴士": ["bus"], "bus": ["bus"],
    "自行": ["bicycle"], "bicycle": ["bicycle"], "bike": ["bicycle"],
    "红绿灯": ["traffic light"], "traffic light": ["traffic light"],
    "消防": ["fire hydrant"], "hydrant": ["fire hydrant"],
    "汽车": ["car", "truck"], "轿车": ["car"], "car": ["car", "truck"],
    "boat": ["boat"], "船": ["boat"],
    "人行横道": ["zebra crossing", "crosswalk", "pedestrian crossing"],
    "crosswalk": ["zebra crossing", "crosswalk", "pedestrian crossing"],
    "楼梯": ["stairway", "stair", "steps"], "stair": ["stairway", "stair", "steps"],
    "桥": ["bridge", "overpass"], "bridge": ["bridge", "overpass"],
    "烟囱": ["chimney", "smokestack"], "chimney": ["chimney", "smokestack"],
    "山": ["mountain", "hill"], "mountain": ["mountain", "hill"],
    "棕榈树": ["palm tree", "palm"], "palm": ["palm tree", "palm"],
    "出租车": ["taxi", "yellow cab"], "taxi": ["taxi", "yellow cab"],
    "拖拉机": ["tractor", "farm vehicle"], "tractor": ["tractor", "farm vehicle"],
    "停车计时器": ["parking meter"], "parking meter": ["parking meter"],
    "自行车": ["bicycle"], "traffic": ["traffic light"],
    "truck": ["truck", "car"], "卡车": ["truck", "car"],
}

# COCO 原生类别（可直接用 YOLO 检测，无需零样本模型）
YOLO_NATIVE_CLASSES = {
    "motorcycle", "bus", "bicycle", "traffic light", "fire hydrant",
    "car", "truck", "boat", "person", "dog", "cat", "bird",
}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_category_info(text_str: str) -> Tuple[List[str], bool, Optional[str]]:
    """
    根据题面文本匹配类别

    :return: (类别列表, 是否需要零样本模型, 命中的关键词)
    """
    text_lower = (text_str or "").lower()
    for keyword, classes in CATEGORY_MAPPING.items():
        if keyword in text_lower:
            use_zero_shot = not all(c in YOLO_NATIVE_CLASSES for c in classes)
            return classes, use_zero_shot, keyword
    return [], False, None


def assess_image_quality(img_bgr) -> dict:
    """
    评估图像质量，检测噪点、马赛克和偏色
    返回各项指标 (0-100，越高越差) 和建议的处理策略
    """
    quality = {
        'noise_level': 0,
        'blockiness': 0,
        'color_cast': 0,
        'strategy': 'full',   # full / gentle / denoise_only / color_correct / none
    }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_diff = np.abs(gray.astype(float) - blur.astype(float))
    noise_level = np.mean(noise_diff)

    if laplacian_var > 500 and noise_level > 10:
        quality['noise_level'] = min(100, int(noise_level * 5))
    elif noise_level > 8:
        quality['noise_level'] = min(100, int(noise_level * 4))

    # 块效应：检测边缘的周期性
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    h, w = gray.shape
    row_edges = np.sum(np.abs(sobel_y), axis=1) / w
    col_edges = np.sum(np.abs(sobel_x), axis=0) / h

    def detect_periodicity(signal, min_period=4, max_period=16):
        if len(signal) < max_period * 3:
            return 0
        std = np.std(signal)
        for period in range(min_period, max_period):
            if len(signal) < period * 3:
                continue
            correlation, count = 0, 0
            for i in range(period, len(signal) - period):
                correlation += abs(signal[i] - signal[i - period])
                count += 1
            if count > 0:
                avg_diff = correlation / count
                if avg_diff < std * 0.5:
                    return min(100, int((1 - avg_diff / (std + 1)) * 100))
        return 0

    quality['blockiness'] = max(detect_periodicity(row_edges), detect_periodicity(col_edges))

    # 偏色：LAB 空间 a/b 通道偏离中性的距离
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    a_mean = np.mean(a) - 128
    b_mean = np.mean(b) - 128
    quality['color_cast'] = min(100, int(np.sqrt(a_mean ** 2 + b_mean ** 2) * 3))

    total = quality['noise_level'] + quality['blockiness'] + quality['color_cast']
    if total < 30:
        quality['strategy'] = 'full'
    elif quality['noise_level'] > 50 or quality['blockiness'] > 50:
        quality['strategy'] = 'denoise_only'
    elif quality['color_cast'] > 40:
        quality['strategy'] = 'color_correct'
    elif total < 80:
        quality['strategy'] = 'gentle'
    else:
        quality['strategy'] = 'none'
    return quality


def preprocess_image(img: Image.Image) -> Image.Image:
    """按图像质量自适应增强"""
    img_np = np.array(img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    quality = assess_image_quality(img_bgr)
    if DEBUG:
        print(f"   📊 图像质量: 噪点={quality['noise_level']}, "
              f"马赛克={quality['blockiness']}, 偏色={quality['color_cast']}, "
              f"策略={quality['strategy']}")
    strategy = quality['strategy']

    if strategy == 'none':
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if strategy == 'denoise_only':
        h_lum = 10 if quality['noise_level'] > 70 else 6
        out = cv2.fastNlMeansDenoisingColored(img_bgr, None, h_lum, h_lum, 7, 21)
        if quality['blockiness'] > 40:
            out = cv2.bilateralFilter(out, 5, 50, 50)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    if strategy == 'color_correct':
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = cv2.add(a, 128 - int(np.mean(a)))
        b = cv2.add(b, 128 - int(np.mean(b)))
        corrected = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        out = cv2.fastNlMeansDenoisingColored(corrected, None, 3, 3, 7, 21)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    if strategy == 'gentle':
        out = cv2.fastNlMeansDenoisingColored(img_bgr, None, 4, 4, 7, 21)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    out = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    out = cv2.filter2D(out, -1, np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]))
    out = cv2.convertScaleAbs(out, alpha=1.1, beta=5)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def crop_image_from_bytes(image_bytes: bytes, crop_box: Tuple[int, int, int, int]) -> Optional[bytes]:
    """按像素框裁剪，返回 JPEG 字节；失败返回 None"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        out = io.BytesIO()
        img.crop(crop_box).save(out, format='JPEG')
        return out.getvalue()
    except Exception:
        return None


class VisionModels:
    """
    检测模型单例管理器

    YOLO11x 负责 COCO 原生类别，GroundingDINO 负责开放词汇类别。
    两者都按需加载，只加载一次。
    """

    _yolo = None
    _grounding_processor = None
    _grounding_model = None

    @classmethod
    def load(cls, with_grounding: bool = True) -> None:
        """加载模型。with_grounding=False 时跳过零样本模型（省内存）"""
        if cls._yolo is None:
            print(f"🚀 正在加载 YOLO 模型: {YOLO_WEIGHTS}")
            cls._yolo = YOLO(YOLO_WEIGHTS)
            print("✅ YOLO 加载完成")
        if with_grounding and cls._grounding_processor is None:
            print(f"🚀 正在加载零样本检测模型: {GROUNDING_MODEL_ID}")
            cls._grounding_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
            cls._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID)
            cls._grounding_model.eval()
            print("✅ 零样本检测模型加载完成")

    @classmethod
    def get_yolo(cls):
        if cls._yolo is None:
            cls.load(with_grounding=False)
        return cls._yolo

    @classmethod
    def detect_zero_shot(cls, image: Image.Image, text_prompt: str,
                         threshold: float = GROUNDING_DINO_CONFIDENCE
                         ) -> List[Tuple[List[float], float, str]]:
        """
        开放词汇检测

        :return: [(box_xyxy, score, label), ...]
        """
        if cls._grounding_processor is None:
            cls.load()
        try:
            inputs = cls._grounding_processor(images=image, text=text_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = cls._grounding_model(**inputs)
            results = cls._grounding_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results.get("text_labels", results.get("labels", ["object"] * len(boxes)))
            return [(b.tolist(), float(s), str(l)) for b, s, l in zip(boxes, scores, labels)]
        except Exception as e:
            if DEBUG:
                print(f"   ⚠️ 零样本检测出错: {e}")
            return []

    @classmethod
    def detect_yolo(cls, image: Image.Image, target_classes: List[str],
                    verbose: bool = False) -> List[Tuple[str, float, List[float]]]:
        """
        按类别做 COCO 目标检测

        :return: [(类别名, 置信度, box_xyxy), ...]
        """
        if cls._yolo is None:
            cls.load(with_grounding=False)
        found = []
        for r in cls._yolo(image, verbose=verbose):
            for box in r.boxes:
                cls_name = cls._yolo.names[int(box.cls[0])]
                if cls_name in target_classes:
                    found.append((cls_name, float(box.conf[0]), box.xyxy[0].tolist()))
        return found
