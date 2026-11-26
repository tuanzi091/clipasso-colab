#!/usr/bin/env python
"""
CLIPasso 图像处理固定流程脚本
流程：
1. 使用 u2net_enhanced_best.pth 提取前景（去除背景）
2. 将前景图像作为输入进行抽象简笔画处理
3. 将所有结果保存在一个文件夹中

使用方法：
    python process_image.py <image_path> [--output_dir <dir>] [--num_paths <n>] [--num_iter <n>]
    
示例：
    python process_image.py target_images/cat.png
    python process_image.py target_images/horse.png --num_paths 32 --num_iter 500
"""
import sys
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'diffvg'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import skimage.io
from skimage.transform import resize

from U2Net_.saved_models.u2net_enhanced import U2NETEnhanced
from sketch_utils import get_mask_u2net
import cv2


def calculate_image_complexity(image):
    """
    增强版图像复杂度评估，用于动态确定笔触数量
    
    方法：结合多层次特征进行综合评估
    - 边缘密度：使用 Canny 边缘检测计算边缘像素比例
    - 梯度方差：计算图像梯度的方差
    - 颜色多样性：RGB图像的颜色熵
    - 高频能量：DCT变换后的高频分量能量
    - 空间熵：局部灰度变化的信息熵
    - 语义加权：人脸检测加权（人脸需要更多笔触）
    
    返回：复杂度分数 (0-1 之间)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. 边缘密度（0-1）
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # 2. 梯度方差（归一化到0-1）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_variance = np.var(gradient_magnitude) / (255.0 * 10)  # 调整归一化
    
    # 3. 颜色多样性（RGB图像的颜色熵）
    if len(image.shape) == 3:
        color_entropy = np.mean([
            -np.sum(p * np.log2(p + 1e-10)) 
            for p in [np.histogram(image[..., i], bins=256, range=(0,256))[0]/image.size 
                     for i in range(3)]
        ]) / 8.0  # 归一化到0-1（最大熵为8）
    else:
        color_entropy = 0.1  # 灰度图给予基础值
    
    # 4. 高频能量（DCT变换后的高频分量）
    dct = cv2.dct(np.float32(gray) / 255.0)
    h, w = dct.shape
    high_freq_mask = np.ones_like(dct)
    high_freq_mask[h//4:3*h//4, w//4:3*w//4] = 0  # 低频区域置0
    high_freq_energy = np.sum(np.abs(dct * high_freq_mask)) / (h * w)
    
    # 5. 空间熵（局部灰度变化的信息熵）
    local_var = cv2.GaussianBlur(gray, (11, 11), 0)
    local_var = cv2.Laplacian(local_var, cv2.CV_64F, ksize=3)
    local_var_hist, _ = np.histogram(np.abs(local_var), bins=64)
    local_var_prob = local_var_hist / np.sum(local_var_hist)
    spatial_entropy = -np.sum(local_var_prob * np.log2(local_var_prob + 1e-10)) / 6.0  # 归一化
    
    # 6. 语义加权因子（人脸检测）
    face_weight = _detect_face_and_weight(image)
    
    # 综合评分（加权融合）
    weights = {
        'edge': 0.25,
        'gradient': 0.15,
        'color': 0.20,
        'high_freq': 0.20,
        'spatial_entropy': 0.20
    }
    
    base_complexity = (
        weights['edge'] * edge_density +
        weights['gradient'] * min(gradient_variance, 1.0) +
        weights['color'] * color_entropy +
        weights['high_freq'] * min(high_freq_energy * 10, 1.0) +
        weights['spatial_entropy'] * spatial_entropy
    )
    
    # 应用语义加权
    final_complexity = min(base_complexity * face_weight, 1.0)
    
    return final_complexity


def _detect_face_and_weight(image):
    """
    检测人脸并返回加权因子
    
    返回：加权因子，人脸返回1.0-2.0，非人脸返回1.0
    """
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            face_area = sum(w * h for _, _, w, h in faces)
            image_area = gray.shape[0] * gray.shape[1]
            face_ratio = min(face_area / image_area, 0.5)
            return 1.0 + face_ratio * 2.0
        return 1.0
    except Exception:
        return 1.0


def calculate_dynamic_strokes(complexity, min_strokes=8, max_strokes=32):
    """
    根据图像复杂度动态计算笔触数量（适用于简笔画）
    
    简笔画原则：以少胜多，用简洁的线条表达对象本质
    - 简单图像：4-12 笔触
    - 中等复杂度：12-20 笔触
    - 复杂图像：20-32 笔触
    
    参数：
        complexity: 图像复杂度 (0-1)
        min_strokes: 最小笔触数量（默认8）
        max_strokes: 最大笔触数量（默认32）
    
    返回：建议的笔触数量
    """
    # 使用对数映射，让复杂度增长时笔触增长更平缓
    # 简笔画不需要太多笔触，重点是捕捉主要特征
    adjusted_complexity = np.log1p(complexity) / np.log1p(1.0)  # 对数归一化
    
    # 使用更保守的映射曲线
    num_strokes = int(min_strokes + (max_strokes - min_strokes) * adjusted_complexity)
    
    # 确保笔触数量为偶数（便于对称处理）
    if num_strokes % 2 != 0:
        num_strokes += 1
    
    return num_strokes


def select_curve_type(complexity):
    """
    根据图像复杂度选择最优曲线类型（可学习参数曲线）
    
    曲线类型说明：
    - line (直线): 2个控制点，适合简单形状、几何图形
    - quadratic (二次贝塞尔): 3个控制点，适合平滑曲线、简单轮廓
    - cubic (三次贝塞尔): 4个控制点，适合复杂曲线、精细细节
    - bspline (B样条): 平滑曲线，局部控制
    - catmull_rom (Catmull-Rom): 超顺滑插值曲线
    - arc (圆弧): 几何圆弧曲线
    - variable_width (可变宽手绘): 模拟人手绘制效果
    
    参数：
        complexity: 图像复杂度 (0-1)
    
    返回：
        curve_type: 曲线类型名称
        control_points_per_seg: 每段的控制点数量
    """
    # 根据复杂度选择曲线类型
    if complexity < 0.15:
        # 极简单图像：使用直线
        curve_type = 'line'
        control_points_per_seg = 2
    elif complexity < 0.3:
        # 简单图像：使用圆弧曲线
        curve_type = 'arc'
        control_points_per_seg = 4
    elif complexity < 0.45:
        # 中等复杂度：使用二次贝塞尔曲线
        curve_type = 'quadratic_bezier'
        control_points_per_seg = 3
    elif complexity < 0.6:
        # 中高复杂度：使用三次贝塞尔曲线
        curve_type = 'cubic_bezier'
        control_points_per_seg = 4
    elif complexity < 0.75:
        # 复杂图像：使用 B样条曲线
        curve_type = 'bspline'
        control_points_per_seg = 4
    elif complexity < 0.9:
        # 非常复杂：使用 Catmull-Rom 超顺滑曲线
        curve_type = 'catmull_rom'
        control_points_per_seg = 4
    else:
        # 极复杂：使用可变宽手绘曲线
        curve_type = 'variable_width'
        control_points_per_seg = 4
    
    return curve_type, control_points_per_seg


def select_drawing_style(complexity):
    """
    根据图像复杂度选择最优绘画风格
    
    绘画风格说明：
    - pencil (铅笔): 柔和细腻，灰色调，适合简单图像
    - pen (钢笔): 精确锐利，深色线条，适合中等复杂度
    - brush (毛笔): 笔触变化大，墨水效果，适合复杂图像
    - marker (马克笔): 色彩鲜明，笔触粗犷
    - watercolor (水彩): 透明柔和，色彩叠加
    - woodcut (木刻): 强烈对比，黑白高对比
    - crayon (蜡笔): 粗粝质感，色彩浓厚
    - charcoal (炭笔): 柔和渐变，黑白艺术
    
    参数：
        complexity: 图像复杂度 (0-1)
    
    返回：
        style: 绘画风格名称
        style_params: 风格参数字典
    """
    # 风格参数配置
    style_configs = {
        'pencil': {
            'name': '铅笔 (Pencil)',
            'description': '柔和细腻的铅笔线条，灰色调，适合素描和草图',
            'stroke_width_range': (1.0, 1.8),
            'stroke_width_var': 0.3,
            'opacity_range': (0.8, 1.0),
            'color_type': 'grayscale',
            'noise_level': 0.15,
            'texture': 'soft',
            'line_quality': 'light',
            'curve_type': 'smooth'
        },
        'pen': {
            'name': '钢笔 (Pen)',
            'description': '精确锐利的钢笔线条，深色，适合工程图和技术素描',
            'stroke_width_range': (1.0, 2.0),
            'stroke_width_var': 0.1,
            'opacity_range': (0.9, 1.0),
            'color_type': 'dark',
            'noise_level': 0.02,
            'texture': 'sharp',
            'line_quality': 'precise',
            'curve_type': 'precise'
        },
        'brush': {
            'name': '毛笔 (Brush)',
            'description': '笔触变化丰富的毛笔效果，墨水浓淡，适合中国画和书法',
            'stroke_width_range': (1.2, 2.4),
            'stroke_width_var': 0.6,
            'opacity_range': (0.8, 1.0),
            'color_type': 'ink',
            'noise_level': 0.25,
            'texture': 'flowing',
            'line_quality': 'expressive',
            'curve_type': 'organic'
        },
        'marker': {
            'name': '马克笔 (Marker)',
            'description': '色彩鲜明的马克笔效果，笔触粗犷，适合快速表达',
            'stroke_width_range': (1.6, 2.4),
            'stroke_width_var': 0.2,
            'opacity_range': (0.85, 1.0),
            'color_type': 'vibrant',
            'noise_level': 0.08,
            'texture': 'bold',
            'line_quality': 'thick',
            'curve_type': 'smooth'
        },
        'watercolor': {
            'name': '水彩 (Watercolor)',
            'description': '透明柔和的水彩效果，色彩叠加，适合艺术创作',
            'stroke_width_range': (1.0, 1.8),
            'stroke_width_var': 0.4,
            'opacity_range': (0.8, 0.95),
            'color_type': 'soft_color',
            'noise_level': 0.3,
            'texture': 'blended',
            'line_quality': 'soft',
            'curve_type': 'flowing'
        },
        'woodcut': {
            'name': '木刻 (Woodcut)',
            'description': '强烈对比的木刻效果，黑白高对比，适合版画风格',
            'stroke_width_range': (1.4, 2.2),
            'stroke_width_var': 0.5,
            'opacity_range': (0.95, 1.0),
            'color_type': 'monochrome',
            'noise_level': 0.1,
            'texture': 'rough',
            'line_quality': 'bold',
            'curve_type': 'angular'
        },
        'crayon': {
            'name': '蜡笔 (Crayon)',
            'description': '粗粝质感的蜡笔效果，色彩浓厚，适合儿童画和手绘',
            'stroke_width_range': (1.4, 2.4),
            'stroke_width_var': 0.45,
            'opacity_range': (0.85, 1.0),
            'color_type': 'rich',
            'noise_level': 0.35,
            'texture': 'rough',
            'line_quality': 'textured',
            'curve_type': 'handdrawn'
        },
        'charcoal': {
            'name': '炭笔 (Charcoal)',
            'description': '柔和渐变的炭笔效果，黑白艺术，适合肖像和风景',
            'stroke_width_range': (1.2, 2.2),
            'stroke_width_var': 0.5,
            'opacity_range': (0.85, 1.0),
            'color_type': 'monochrome',
            'noise_level': 0.4,
            'texture': 'soft_blend',
            'line_quality': 'expressive',
            'curve_type': 'smooth'
        }
    }
    
    # 根据复杂度选择风格
    if complexity < 0.15:
        style = 'pencil'
    elif complexity < 0.3:
        style = 'charcoal'
    elif complexity < 0.45:
        style = 'pen'
    elif complexity < 0.55:
        style = 'marker'
    elif complexity < 0.65:
        style = 'brush'
    elif complexity < 0.75:
        # 水彩效果不佳，暂时从自动选择中移除（仍可用 --style watercolor 手动指定）
        style = 'brush'
    elif complexity < 0.85:
        style = 'crayon'
    else:
        style = 'woodcut'
    
    return style, style_configs[style]


def get_drawing_style_params(style):
    """
    获取指定绘画风格的完整参数
    
    参数：
        style: 绘画风格名称
    
    返回：
        完整的风格参数字典
    """
    style_configs = {
        'pencil': {
            'name': '铅笔 (Pencil)',
            'description': '柔和细腻的铅笔线条，灰色调，适合素描和草图',
            'stroke_width_range': (1.0, 1.8),
            'stroke_width_var': 0.3,
            'opacity_range': (0.8, 1.0),
            'color_type': 'grayscale',
            'noise_level': 0.15,
            'texture': 'soft',
            'line_quality': 'light',
            'curve_type': 'smooth'
        },
        'pen': {
            'name': '钢笔 (Pen)',
            'description': '精确锐利的钢笔线条，深色，适合工程图和技术素描',
            'stroke_width_range': (1.0, 2.0),
            'stroke_width_var': 0.1,
            'opacity_range': (0.9, 1.0),
            'color_type': 'dark',
            'noise_level': 0.02,
            'texture': 'sharp',
            'line_quality': 'precise',
            'curve_type': 'precise'
        },
        'brush': {
            'name': '毛笔 (Brush)',
            'description': '笔触变化丰富的毛笔效果，墨水浓淡，适合中国画和书法',
            'stroke_width_range': (1.2, 2.4),
            'stroke_width_var': 0.6,
            'opacity_range': (0.8, 1.0),
            'color_type': 'ink',
            'noise_level': 0.25,
            'texture': 'flowing',
            'line_quality': 'expressive',
            'curve_type': 'organic'
        },
        'marker': {
            'name': '马克笔 (Marker)',
            'description': '色彩鲜明的马克笔效果，笔触粗犷，适合快速表达',
            'stroke_width_range': (1.6, 2.4),
            'stroke_width_var': 0.2,
            'opacity_range': (0.85, 1.0),
            'color_type': 'vibrant',
            'noise_level': 0.08,
            'texture': 'bold',
            'line_quality': 'thick',
            'curve_type': 'smooth'
        },
        'watercolor': {
            'name': '水彩 (Watercolor)',
            'description': '透明柔和的水彩效果，色彩叠加，适合艺术创作',
            'stroke_width_range': (1.0, 1.8),
            'stroke_width_var': 0.4,
            'opacity_range': (0.8, 0.95),
            'color_type': 'soft_color',
            'noise_level': 0.3,
            'texture': 'blended',
            'line_quality': 'soft',
            'curve_type': 'flowing'
        },
        'woodcut': {
            'name': '木刻 (Woodcut)',
            'description': '强烈对比的木刻效果，黑白高对比，适合版画风格',
            'stroke_width_range': (1.4, 2.2),
            'stroke_width_var': 0.5,
            'opacity_range': (0.95, 1.0),
            'color_type': 'monochrome',
            'noise_level': 0.1,
            'texture': 'rough',
            'line_quality': 'bold',
            'curve_type': 'angular'
        },
        'crayon': {
            'name': '蜡笔 (Crayon)',
            'description': '粗粝质感的蜡笔效果，色彩浓厚，适合儿童画和手绘',
            'stroke_width_range': (1.4, 2.4),
            'stroke_width_var': 0.45,
            'opacity_range': (0.85, 1.0),
            'color_type': 'rich',
            'noise_level': 0.35,
            'texture': 'rough',
            'line_quality': 'textured',
            'curve_type': 'handdrawn'
        },
        'charcoal': {
            'name': '炭笔 (Charcoal)',
            'description': '柔和渐变的炭笔效果，黑白艺术，适合肖像和风景',
            'stroke_width_range': (1.2, 2.2),
            'stroke_width_var': 0.5,
            'opacity_range': (0.85, 1.0),
            'color_type': 'monochrome',
            'noise_level': 0.4,
            'texture': 'soft_blend',
            'line_quality': 'expressive',
            'curve_type': 'smooth'
        }
    }
    return style_configs.get(style, style_configs['pen'])


def get_curve_type_description(curve_type):
    """
    获取曲线类型的描述信息
    
    参数：
        curve_type: 曲线类型名称
    
    返回：
        描述信息字典
    """
    descriptions = {
        'line': {
            'name': '直线',
            'control_points': 2,
            'description': '最简单，只有起点和终点，适合几何图形和简单轮廓',
            'formula': 'P(t) = (1-t)P₀ + tP₁',
            'style': '简洁几何'
        },
        'arc': {
            'name': '圆弧曲线',
            'control_points': 4,
            'description': '几何圆弧，平滑过渡，适合绘制圆形、弧形等几何形状',
            'formula': '基于圆心和半径的圆弧参数方程',
            'style': '几何风格'
        },
        'quadratic_bezier': {
            'name': '二次贝塞尔曲线',
            'control_points': 3,
            'description': '平滑过渡，一个拐点，适合流畅曲线和简单轮廓',
            'formula': 'P(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂',
            'style': '平滑流畅'
        },
        'cubic_bezier': {
            'name': '三次贝塞尔曲线',
            'control_points': 4,
            'description': '更灵活，两个拐点，适合复杂曲线和精细细节',
            'formula': 'P(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃',
            'style': '标准曲线'
        },
        'bspline': {
            'name': 'B样条曲线',
            'control_points': 4,
            'description': '平滑曲线，局部控制能力强，适合绘制光滑的复杂曲线',
            'formula': 'P(t) = ΣBᵢₖ(t)Pᵢ (B样条基函数)',
            'style': '平滑细腻'
        },
        'catmull_rom': {
            'name': 'Catmull-Rom 超顺滑曲线',
            'control_points': 4,
            'description': '插值曲线，通过所有控制点，超顺滑过渡，最自然的曲线',
            'formula': '基于Hermite插值，C¹连续',
            'style': '超顺滑'
        },
        'variable_width': {
            'name': '可变宽手绘曲线',
            'control_points': 4,
            'description': '模拟人手绘制效果，笔画宽度可变化，最具艺术感',
            'formula': '带宽度参数的曲线 + 抖动模拟',
            'style': '手绘风格'
        },
        'rational_bezier': {
            'name': '有理贝塞尔曲线',
            'control_points': 4,
            'description': '带权重的曲线，可精确表示圆锥曲线',
            'formula': 'P(t) = Σ[wᵢBᵢ(t)Pᵢ] / Σ[wᵢBᵢ(t)]',
            'style': '精确曲线'
        }
    }
    return descriptions.get(curve_type, descriptions['cubic_bezier'])


def detect_key_region(image_array, min_size_threshold=224):
    """
    智能检测图像中的关键区域
    
    方法：结合多种检测技术
    1. 人脸检测 - 检测人脸位置（优先级最高）
    2. 显性检测 - 使用边缘密度和梯度分析找出细节丰富的区域
    3. 显性图分析 - 使用简单的显著性算法
    
    参数：
        image_array: 输入图像数组 (RGB)
        min_size_threshold: 小尺寸阈值（小于此值触发智能裁剪）
    
    返回：
        key_region: (x, y, w, h) 关键区域坐标
        region_type: 关键区域类型描述
    """
    height, width = image_array.shape[:2]
    
    # 如果图像尺寸足够大，不需要智能裁剪
    if height >= min_size_threshold and width >= min_size_threshold:
        return None, "图像尺寸足够，无需智能裁剪"
    
    print(f"[Key Region Detection] 图像尺寸 {width}x{height} 小于阈值 {min_size_threshold}")
    
    # 方法1：人脸检测（优先级最高）
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # 找到最大的人脸
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        # 扩展人脸区域，包含更多上下文
        expand_ratio = 1.5
        new_w = int(w * expand_ratio)
        new_h = int(h * expand_ratio)
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        # 确保不超出图像边界
        new_w = min(new_w, width - new_x)
        new_h = min(new_h, height - new_y)
        return (new_x, new_y, new_w, new_h), f"检测到人脸区域 ({len(faces)}个人脸)"
    
    # 方法2：边缘密度分析（找出细节丰富的区域）
    edges = cv2.Canny(gray, 50, 150)
    
    # 将图像划分为网格，计算每个区域的边缘密度
    grid_size = 4
    cell_h = height // grid_size
    cell_w = width // grid_size
    
    max_density = 0
    best_cell = (0, 0, cell_w, cell_h)
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell_y = i * cell_h
            cell_x = j * cell_w
            cell_edges = edges[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w]
            density = np.sum(cell_edges > 0) / (cell_h * cell_w)
            if density > max_density:
                max_density = density
                best_cell = (cell_x, cell_y, cell_w, cell_h)
    
    # 方法3：显著性检测（使用梯度和颜色对比）
    # 计算颜色对比度
    if len(image_array.shape) == 3:
        # 计算每个像素与周围区域的颜色差异
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # 计算局部对比度
        blur = cv2.GaussianBlur(l_channel, (21, 21), 0)
        contrast = np.abs(l_channel - blur)
        
        # 找到对比度最高的区域
        max_contrast_region = find_max_region(contrast, grid_size)
        
        # 结合边缘密度和对比度
        if max_contrast_region is not None:
            # 如果对比度区域与边缘密度区域不同，选择两者中更显著的
            contrast_x, contrast_y, contrast_w, contrast_h = max_contrast_region
            contrast_score = np.mean(contrast[contrast_y:contrast_y+contrast_h, 
                                              contrast_x:contrast_x+contrast_w])
            edge_score = max_density
            
            if contrast_score > edge_score * 100:  # 对比度权重更高
                best_cell = max_contrast_region
    
    return best_cell, f"检测到高细节区域 (边缘密度: {max_density:.4f})"


def find_max_region(array, grid_size=4):
    """在数组中找到平均值最大的区域"""
    height, width = array.shape
    cell_h = height // grid_size
    cell_w = width // grid_size
    
    max_mean = 0
    best_region = None
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell_y = i * cell_h
            cell_x = j * cell_w
            cell_array = array[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w]
            mean_val = np.mean(cell_array)
            if mean_val > max_mean:
                max_mean = mean_val
                best_region = (cell_x, cell_y, cell_w, cell_h)
    
    return best_region


def smart_crop_and_zoom(image, key_region, target_size=224):
    """
    智能裁剪并放大关键区域到目标尺寸
    
    参数：
        image: PIL Image 对象
        key_region: (x, y, w, h) 关键区域坐标
        target_size: 目标尺寸（默认224）
    
    返回：
        cropped_image: 裁剪并放大后的 PIL Image
        crop_info: 裁剪信息字典
    """
    x, y, w, h = key_region
    width, height = image.size
    
    # 裁剪关键区域
    cropped = image.crop((x, y, x + w, y + h))
    
    # 计算放大因子
    scale_factor = target_size / max(w, h)
    
    # 放大到目标尺寸
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)
    resized = cropped.resize((new_width, new_height), Image.BICUBIC)
    
    # 如果不是正方形，进行中心裁剪
    if new_width != target_size or new_height != target_size:
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        # 确保裁剪区域在图像范围内
        left = max(0, left)
        top = max(0, top)
        right = min(new_width, right)
        bottom = min(new_height, bottom)
        resized = resized.crop((left, top, right, bottom))
        
        # 如果裁剪后尺寸不足，进行填充或再次放大
        if resized.size[0] < target_size or resized.size[1] < target_size:
            # 创建目标尺寸的白色背景
            final = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            # 将裁剪后的图像粘贴到中心
            paste_x = (target_size - resized.size[0]) // 2
            paste_y = (target_size - resized.size[1]) // 2
            final.paste(resized, (paste_x, paste_y))
            resized = final
    
    crop_info = {
        'original_size': (width, height),
        'key_region': key_region,
        'scale_factor': scale_factor,
        'final_size': (target_size, target_size)
    }
    
    return resized, crop_info


def process_small_image_smart(image_path, output_dir, target_size=224):
    """
    处理小尺寸图像：智能检测关键部位，放大裁剪到目标尺寸
    
    参数：
        image_path: 输入图像路径
        output_dir: 输出目录
        target_size: 目标尺寸（默认224）
    
    返回：
        processed_image: 处理后的 PIL Image
        was_processed: 是否进行了智能裁剪
    """
    # 读取图像
    image = Image.open(image_path)
    if image.mode == "RGBA":
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, (0, 0), image)
        image = new_image
    image = image.convert("RGB")
    
    width, height = image.size
    
    # 检查是否需要智能裁剪
    if width >= target_size and height >= target_size:
        print(f"[Smart Crop] 图像尺寸 {width}x{height} 足够，无需智能裁剪")
        return image, False
    
    print(f"\n[Smart Crop] 图像尺寸 {width}x{height} 小于目标尺寸 {target_size}")
    print("[Smart Crop] 启动智能关键部位检测...")
    
    # 转换为数组进行检测
    image_array = np.array(image)
    
    # 检测关键区域
    key_region, region_type = detect_key_region(image_array, target_size)
    
    if key_region is None:
        # 如果没有检测到关键区域，直接放大整个图像
        print(f"[Smart Crop] 未检测到特定关键区域，放大整个图像")
        processed = image.resize((target_size, target_size), Image.BICUBIC)
        return processed, True
    
    print(f"[Smart Crop] {region_type}")
    print(f"[Smart Crop] 关键区域坐标: x={key_region[0]}, y={key_region[1]}, w={key_region[2]}, h={key_region[3]}")
    
    # 智能裁剪并放大
    processed, crop_info = smart_crop_and_zoom(image, key_region, target_size)
    
    print(f"[Smart Crop] 放大因子: {crop_info['scale_factor']:.2f}")
    print(f"[Smart Crop] 最终尺寸: {crop_info['final_size']}")
    
    # 保存智能裁剪后的图像
    smart_crop_path = os.path.join(output_dir, "smart_crop.png")
    processed.save(smart_crop_path)
    print(f"[Smart Crop] 保存智能裁剪图像到: {smart_crop_path}")
    
    return processed, True


def extract_foreground(image_path, output_dir):
    """使用U2Net提取前景"""
    target = Image.open(image_path)
    if target.mode == "RGBA":
        new_image = Image.new("RGBA", target.size, "WHITE")
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    
    # 创建临时的args对象
    class TempArgs:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.use_gpu = torch.cuda.is_available()
            self.output_dir = output_dir
    
    temp_args = TempArgs()
    masked_im, mask = get_mask_u2net(args=temp_args, pil_im=target)
    return masked_im, mask


def _mask_to_2d(mask):
    """把 mask 统一转成 2D float32（0~1），兼容 torch 张量 / 多通道 / 0-255。"""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask)
    if m.ndim == 4:
        m = m[0, 0]
    elif m.ndim == 3:
        m = m[0] if m.shape[0] == 1 else m.mean(axis=-1)
    m = m.astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    return m


def _detect_face_bbox(image):
    """用 OpenCV Haar 检测最大人脸，返回 (x, y, w, h, W, H) 或 None。"""
    try:
        import cv2
        img = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        return int(x), int(y), int(w), int(h), img.shape[1], img.shape[0]
    except Exception as e:
        print(f"[Mask] face detect skipped: {e}")
        return None


def refine_foreground_mask(mask, image=None, use_grabcut=True, dilate_iters=2):
    """掩码精修：GrabCut 收紧 -> 最大连通域 -> 填洞 -> 轻微膨胀。

    针对暗背景、低对比人像：U2Net 常把大范围背景判为前景（前景占比过高），
    这里用图像颜色做一次 GrabCut，再用形态学保证前景连续、无碎点。
    """
    try:
        m = _mask_to_2d(mask)
    except Exception as e:
        print(f"[Mask] normalize failed: {e}")
        return mask, False

    if m.size == 0:
        return mask, False

    h, w = m.shape

    # 1) GrabCut 二次分割（以 U2Net 结果为初始标记）
    if image is not None and use_grabcut:
        try:
            import cv2
            img = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
            if img.shape[:2] != (h, w):
                m_pil = Image.fromarray((m * 255).astype(np.uint8))
                m_pil = m_pil.resize((img.shape[1], img.shape[0]), Image.BICUBIC)
                m = np.array(m_pil).astype(np.float32) / 255.0
                h, w = m.shape
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
            gc[m > 0.85] = cv2.GC_FGD
            gc[m < 0.15] = cv2.GC_BGD
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
            if out.mean() > 0.01:
                m = out
        except Exception as e:
            print(f"[Mask] grabcut skipped: {e}")

    # 2) 最大连通域 + 填洞 + 膨胀
    try:
        from scipy import ndimage as ndi
        binary = m > 0.5
        if binary.sum() == 0:
            return mask, False
        labels, num = ndi.label(binary)
        if num > 1:
            sizes = ndi.sum(binary, labels, range(1, num + 1))
            binary = labels == (int(np.argmax(sizes)) + 1)
        binary = ndi.binary_fill_holes(binary)

        # 3) 前景占比过高时，用扩展人脸椭圆硬性约束人像范围
        if image is not None and float(binary.mean()) > 0.6:
            fb = _detect_face_bbox(image)
            if fb is not None:
                fx, fy, fw, fh, W, H = fb
                if binary.shape != (H, W):
                    b_pil = Image.fromarray((binary * 255).astype(np.uint8))
                    b_pil = b_pil.resize((W, H), Image.BICUBIC)
                    binary = np.array(b_pil) > 127
                cx = fx + fw / 2.0
                cy = fy + fh / 2.0
                rx = max(fw * 1.8, W * 0.32)
                ry = max(fh * 2.8, H * 0.48)
                yy, xx = np.mgrid[0:H, 0:W]
                ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
                cand = binary & ellipse
                if cand.mean() > 0.02:
                    binary = cand
                    print(f"[Mask] face-box constraint applied: ratio={binary.mean():.3f}")

        binary = ndi.binary_fill_holes(binary)
        binary = ndi.binary_dilation(binary, iterations=dilate_iters)
        print(f"[Mask] refined: foreground ratio={binary.mean():.3f}")
        return binary.astype(np.float32), True
    except Exception as e:
        print(f"[Mask] refine skipped: {e}")
        return mask, False


def process_image(image_path, output_dir=None, num_paths=32, num_iter=500, save_interval=50, use_gpu=False, crop_square=False, dynamic_strokes=True, dynamic_curve=True, curve_type=None, drawing_style=None, face_focus=0.0, image_scale=None, stroke_width=None, mask_penalty=0.0):
    """
    按固定流程处理图像（默认启用动态自动选择）
    
    参数：
        image_path: 输入图像路径
        output_dir: 输出目录（默认为 results/<图像名>）
        num_paths: 笔触数量（默认32），动态模式下作为最大值
        num_iter: 迭代次数（默认500）
        save_interval: 保存间隔（默认50）
        curve_type: 指定曲线类型（None表示动态选择）
        drawing_style: 指定绘画风格（None表示动态选择）
        use_gpu: 是否使用GPU（默认False）
        crop_square: 是否裁剪为正方形（保持原始分辨率）
        dynamic_strokes: 是否使用动态笔触数量（默认True，基于图像复杂度）
        dynamic_curve: 是否使用动态曲线类型（默认True，基于图像复杂度）
    """
    # 获取图像基本信息
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join("results", image_name)
    
    print("=" * 60)
    print(f"Processing: {image_name}")
    print("=" * 60)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\n[Step 1] Created output directory: {output_dir}")
    
    # 检查图像尺寸，对小尺寸图像进行智能裁剪
    target = Image.open(image_path)
    if target.mode == "RGBA":
        new_image = Image.new("RGBA", target.size, "WHITE")
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    
    # 获取图像尺寸
    width, height = target.size
    print(f"[Image Info] Original Size: {width}x{height}")
    
    # 智能裁剪处理（对小尺寸图像）
    smart_crop_enabled = True  # 默认启用智能裁剪
    target_scale = int(image_scale) if image_scale else 224
    
    if width < target_scale or height < target_scale:
        print(f"\n[Step 2] Smart Crop - 小尺寸图像检测")
        # 创建临时文件用于智能裁剪处理
        temp_original_path = os.path.join(output_dir, "original_before_smart_crop.png")
        target.save(temp_original_path)
        
        # 执行智能裁剪
        processed_image, was_cropped = process_small_image_smart(temp_original_path, output_dir, target_scale)
        
        if was_cropped:
            # 使用智能裁剪后的图像继续处理
            target = processed_image
            width, height = target.size
            print(f"[Image Info] After Smart Crop: {width}x{height}")
            
            # 保存智能裁剪后的图像作为"原始图像"
            smart_crop_final_path = os.path.join(output_dir, "original_smart_cropped.png")
            target.save(smart_crop_final_path)
            
            # 更新image_path为智能裁剪后的图像路径（用于后续前景提取）
            image_path = smart_crop_final_path
        else:
            # 如果没有进行智能裁剪，直接放大
            target = target.resize((target_scale, target_scale), Image.BICUBIC)
            width, height = target.size
            print(f"[Image Info] After Resize: {width}x{height}")
            
            resized_path = os.path.join(output_dir, "original_resized.png")
            target.save(resized_path)
            image_path = resized_path
    
    # 计算图像缩放尺寸（当crop_square时裁剪为正方形）
    if crop_square and width >= target_scale and height >= target_scale:
        # 裁剪为正方形，使用最小边作为尺寸
        min_size = min(width, height)
        target_scale = min_size
        print(f"[Crop Square] Cropping to square: {target_scale}x{target_scale}")
        # 居中裁剪
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        target = target.crop((left, top, right, bottom))
        print(f"[Crop Square] Cropped from ({left},{top}) to ({right},{bottom})")
    
    original_path = os.path.join(output_dir, "original.png")
    target.save(original_path)
    print(f"\n[Step 3] Saved original image to {original_path}")
    
    # 提取前景
    print("\n[Step 4] Extracting foreground with u2net_enhanced_best.pth...")
    masked_im, mask = extract_foreground(image_path, output_dir)

    # 掩码后处理：GrabCut 收紧 + 去碎点 + 填洞 + 轻微膨胀，让笔触更聚焦前景
    refined_mask, mask_ok = refine_foreground_mask(mask, image=target)
    if mask_ok:
        rgb = np.array(target.convert("RGB")).astype(np.float32)
        m_np = refined_mask
        if m_np.shape[:2] != rgb.shape[:2]:
            m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
            m_pil = m_pil.resize((rgb.shape[1], rgb.shape[0]), Image.BICUBIC)
            m_np = np.array(m_pil).astype(np.float32) / 255.0
        alpha = m_np[..., None]
        comp = rgb * alpha + 255.0 * (1.0 - alpha)
        masked_im = Image.fromarray(comp.clip(0, 255).astype(np.uint8))
        mask = refined_mask
        print("[Step 4] Mask refined: largest component + fill holes + dilate")
    
    # 保存前景图
    foreground_path = os.path.join(output_dir, "foreground.png")
    masked_im.save(foreground_path)
    print(f"[Step 5] Saved foreground image to {foreground_path}")
    
    # 动态计算笔触数量和曲线类型
    if dynamic_strokes or dynamic_curve:
        print("\n[Step 6] Calculating image complexity...")
        # 使用前景图像计算复杂度
        foreground_array = np.array(masked_im)
        complexity = calculate_image_complexity(foreground_array)
        print(f"[Complexity] Score: {complexity:.4f} (0=简单, 1=复杂)")
        
        # 动态计算笔触数量
        if dynamic_strokes:
            num_paths = calculate_dynamic_strokes(complexity, min_strokes=20, max_strokes=num_paths)
            print(f"[Dynamic Strokes] Using {num_paths} strokes based on image complexity")
        
        # 曲线类型选择：优先使用手动指定，其次动态选择
        if curve_type is not None:
            # 手动指定曲线类型
            curve_desc = get_curve_type_description(curve_type)
            control_points_per_seg = curve_desc['control_points']
            print(f"[Curve Type] Manually specified: {curve_desc['name']} ({curve_desc['style']})")
        elif dynamic_curve:
            # 动态选择曲线类型
            curve_type, control_points_per_seg = select_curve_type(complexity)
            curve_desc = get_curve_type_description(curve_type)
            print(f"[Dynamic Curve] Using {curve_desc['name']} ({curve_desc['style']}) with {control_points_per_seg} control points")
            print(f"  Formula: {curve_desc['formula']}")
            print(f"  Description: {curve_desc['description']}")
        else:
            curve_type = 'cubic_bezier'
            control_points_per_seg = 4
            print(f"[Curve Type] Using fixed: cubic_bezier (三次贝塞尔曲线)")
        
        # 绘画风格选择：优先使用手动指定，其次动态选择
        if drawing_style is not None:
            # 手动指定绘画风格
            style_params = get_drawing_style_params(drawing_style)
            print(f"[Drawing Style] Manually specified: {style_params['name']}")
            print(f"  Width Range: {style_params['stroke_width_range']}")
            print(f"  Opacity Range: {style_params['opacity_range']}")
            print(f"  Description: {style_params['description']}")
        else:
            # 动态选择绘画风格
            drawing_style, style_params = select_drawing_style(complexity)
            print(f"[Drawing Style] Using {style_params['name']} based on image complexity")
            print(f"  Width Range: {style_params['stroke_width_range']}")
            print(f"  Opacity Range: {style_params['opacity_range']}")
            print(f"  Description: {style_params['description']}")
    else:
        print(f"\n[Step 6] Using fixed stroke count: {num_paths}")
        curve_type = 'cubic_bezier'
        control_points_per_seg = 4
        style_params = None
    
    # 保存mask
    mask_path = os.path.join(output_dir, "mask.png")
    if mask is not None:
        # 将mask转换为PIL图像保存
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        if mask_np.ndim == 4:
            mask_np = mask_np.squeeze()
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if mask_np.ndim == 3:
            mask_np = mask_np.mean(axis=-1)
        # 确保mask在0-255范围内
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)
        mask_pil.save(mask_path)
        print(f"[Step 7] Saved mask to {mask_path}")
    
    # 运行painterly_rendering
    # 使用前景图像作为输入，并传递我们提取的mask
    print("\n[Step 8] Generating sketch...")
    from subprocess import run as sp_run
    
    cmd = [
        sys.executable, 
        os.path.join(REPO_ROOT, 'painterly_rendering.py'),
        foreground_path,  # 使用前景图像作为输入
        '--output_dir', output_dir,
        '--wandb_name', image_name,
        '--num_paths', str(num_paths),
        '--num_iter', str(num_iter),
        '--save_interval', str(save_interval),
        '--use_wandb', '0',
        '--use_gpu', str(int(use_gpu)),
        '--image_scale', str(target_scale),  # 使用计算的图像尺寸
        '--external_mask', mask_path,     # 使用外部mask
        '--mask_object_attention', '1',   # 将mask应用到注意力图，确保特征点只落在前景区域
        '--control_points_per_seg', str(control_points_per_seg),  # 控制点数量（曲线阶数）
        '--curve_type', str(curve_type)   # 曲线类型：真正决定笔触初始化几何
    ]
    # 面部加权：提高五官区域的笔触占比
    if face_focus and face_focus > 0:
        cmd.extend(['--face_emphasis', str(face_focus)])
    # 固定线宽（用于更细致的刻画）
    if stroke_width and stroke_width > 0:
        cmd.extend(['--width', str(stroke_width)])
    # 笔触越界惩罚
    if mask_penalty and mask_penalty > 0:
        cmd.extend(['--mask_penalty', str(mask_penalty)])
    # 绘画风格落地：逐笔触采样线宽与透明度
    if style_params is not None:
        opacity_range = style_params['opacity_range']
        if not (stroke_width and stroke_width > 0):
            width_range = style_params['stroke_width_range']
            cmd.extend([
                '--stroke_width_min', str(width_range[0]),
                '--stroke_width_max', str(width_range[1]),
            ])
        cmd.extend([
            '--opacity_min', str(opacity_range[0]),
            '--opacity_max', str(opacity_range[1]),
        ])
    
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(REPO_ROOT, 'diffvg')
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print(f"Running: {' '.join(cmd)}")
    result = sp_run(cmd, env=env, cwd=REPO_ROOT, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ Successfully processed!")
        print("=" * 60)
        print(f"\nAll outputs saved in {output_dir}:")
        print("- original.png          (原始图像)")
        print("- foreground.png        (提取的前景)")
        print("- mask.png              (前景mask)")
        print("- best_iter.jpg/svg     (最佳简笔画)")
        print("- final_svg.svg         (最终简笔画)")
        print("- jpg_logs/             (迭代过程JPG)")
        print("- svg_logs/             (迭代过程SVG)")
    else:
        print("\n" + "=" * 60)
        print("✗ Failed to process")
        print("=" * 60)
        if result.stderr:
            print("Error:", result.stderr[-1000:])
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='CLIPasso 图像处理固定流程：前景提取 + 简笔画生成（默认启用动态自动选择）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python process_image.py target_images/cat.png         # 默认：动态笔触 + 动态曲线 + 动态风格
    python process_image.py target_images/horse.png       # 自动根据图像复杂度选择参数
    python process_image.py target_images/rose.png --no_dynamic_strokes  # 禁用动态笔触
    python process_image.py target_images/rose.png --no_dynamic_curve    # 禁用动态曲线
    python process_image.py target_images/rose.png --curve_type bspline  # 手动指定曲线类型
    python process_image.py target_images/cat.png --style pencil        # 手动指定绘画风格
    python process_image.py target_images/cat.png --num_paths 48         # 动态模式下最大笔触数
    
动态曲线类型映射（根据图像复杂度自动选择）:
    复杂度 < 0.15  → 直线 (line)
    0.15 - 0.30   → 圆弧曲线 (arc)
    0.30 - 0.45   → 二次贝塞尔 (quadratic_bezier)
    0.45 - 0.60   → 三次贝塞尔 (cubic_bezier)
    0.60 - 0.75   → B样条曲线 (bspline)
    0.75 - 0.90   → Catmull-Rom超顺滑曲线 (catmull_rom)
    复杂度 ≥ 0.90 → 可变宽手绘曲线 (variable_width)
        """
    )
    
    parser.add_argument('image_path', help='输入图像路径')
    parser.add_argument('--output_dir', help='输出目录（默认为 results/<图像名>）')
    parser.add_argument('--num_paths', type=int, default=32, help='笔触数量（默认32），动态模式下为最大值')
    parser.add_argument('--num_iter', type=int, default=500, help='迭代次数（默认500）')
    parser.add_argument('--save_interval', type=int, default=50, help='保存间隔（默认50）')
    parser.add_argument('--use_gpu', type=int, default=0, help='是否使用GPU（0=否，1=是）')
    parser.add_argument('--crop_square', action='store_true', help='裁剪为正方形（保持原始分辨率）')
    parser.add_argument('--no_dynamic_strokes', action='store_true', help='禁用动态笔触数量（使用固定值）')
    parser.add_argument('--no_dynamic_curve', action='store_true', help='禁用动态曲线类型（使用三次贝塞尔）')
    parser.add_argument('--curve_type', type=str, default=None, 
                        choices=['line', 'arc', 'quadratic_bezier', 'cubic_bezier', 
                                'bspline', 'catmull_rom', 'variable_width', 'rational_bezier'],
                        help='手动指定曲线类型（覆盖动态选择）：\n'
                             '  line - 直线（简洁几何）\n'
                             '  arc - 圆弧曲线（几何风格）\n'
                             '  quadratic_bezier - 二次贝塞尔（平滑流畅）\n'
                             '  cubic_bezier - 三次贝塞尔（标准曲线）\n'
                             '  bspline - B样条曲线（平滑细腻）\n'
                             '  catmull_rom - Catmull-Rom超顺滑曲线\n'
                             '  variable_width - 可变宽手绘曲线（最像人画）\n'
                             '  rational_bezier - 有理贝塞尔曲线')
    parser.add_argument('--style', type=str, default=None,
                        choices=['pencil', 'pen', 'brush', 'marker', 'watercolor', 
                                'woodcut', 'crayon', 'charcoal'],
                        help='手动指定绘画风格（覆盖动态选择）：\n'
                             '  pencil - 铅笔（柔和细腻，灰色调）\n'
                             '  charcoal - 炭笔（柔和渐变，黑白艺术）\n'
                             '  pen - 钢笔（精确锐利，深色线条）\n'
                             '  marker - 马克笔（色彩鲜明，粗犷笔触）\n'
                             '  brush - 毛笔（笔触变化，墨水效果）\n'
                             '  watercolor - 水彩（透明柔和，色彩叠加）\n'
                             '  crayon - 蜡笔（粗粝质感，色彩浓厚）\n'
                             '  woodcut - 木刻（强烈对比，黑白高对比）')
    parser.add_argument('--face_focus', type=float, default=0.0,
                        help='面部加权强度（默认0=关闭）。>0 时提高五官区域的笔触初始化权重，'
                             '减少头发/背景的笔触，人像推荐 1.0~2.0')
    parser.add_argument('--image_scale', type=int, default=None,
                        help='画布/输入尺寸（默认224）。提到 320/384 可获得更细致的刻画，速度按平方变慢')
    parser.add_argument('--stroke_width', type=float, default=None,
                        help='固定线宽（默认走风格自适应）。小于1.0 可画出更细的线条')
    parser.add_argument('--mask_penalty', type=float, default=0.0,
                        help='笔触越界惩罚系数（默认0关闭）。>0 时惩罚落在 mask 外的笔触，'
                             '推荐 5~20，可显著减少背景笔触')
    
    args = parser.parse_args()
    
    # 检查图像是否存在
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # 处理图像（默认启用动态自动选择）
    success = process_image(
        image_path=args.image_path,
        output_dir=args.output_dir,
        num_paths=args.num_paths,
        num_iter=args.num_iter,
        save_interval=args.save_interval,
        use_gpu=bool(args.use_gpu),
        crop_square=args.crop_square,
        dynamic_strokes=not args.no_dynamic_strokes,
        dynamic_curve=not args.no_dynamic_curve,
        curve_type=args.curve_type,
        drawing_style=args.style,
        face_focus=args.face_focus,
        image_scale=args.image_scale,
        stroke_width=args.stroke_width,
        mask_penalty=args.mask_penalty
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()