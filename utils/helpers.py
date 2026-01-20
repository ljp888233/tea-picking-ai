"""
辅助工具函数
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def calculate_angle(point1, point2, point3):
    """
    计算三个点形成的角度
    point2 是角的顶点
    返回角度（0-180度）
    """
    a = np.array([point1.x, point1.y])
    b = np.array([point2.x, point2.y])
    c = np.array([point3.x, point3.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle


def calculate_distance(point1, point2):
    """
    计算两个点之间的欧氏距离（归一化坐标）
    """
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def draw_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文文字
    """
    # 转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)  # 微软雅黑
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)  # 黑体
        except:
            font = ImageFont.load_default()
    
    # 绘制文字
    draw.text(position, text, font=font, fill=color[::-1])  # BGR转RGB
    
    # 转回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_landmark_coords(landmark, frame_shape):
    """
    将归一化的landmark坐标转换为像素坐标
    """
    h, w = frame_shape[:2]
    return int(landmark.x * w), int(landmark.y * h)


def smooth_value(current, previous, alpha=0.3):
    """
    平滑数值，减少抖动
    alpha: 平滑系数，越小越平滑
    """
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


def get_score_color(score):
    """
    根据分数返回颜色 (BGR格式)
    """
    if score >= 80:
        return (0, 255, 0)    # 绿色 - 优秀
    elif score >= 60:
        return (0, 255, 255)  # 黄色 - 良好
    elif score >= 40:
        return (0, 165, 255)  # 橙色 - 一般
    else:
        return (0, 0, 255)    # 红色 - 需改进


def get_score_level(score):
    """
    根据分数返回等级称号
    """
    if score >= 90:
        return "采茶大师 🏆"
    elif score >= 80:
        return "采茶高手 ⭐"
    elif score >= 70:
        return "采茶能手 👍"
    elif score >= 60:
        return "采茶学徒 📚"
    elif score >= 40:
        return "采茶新手 🌱"
    else:
        return "初来乍到 👶"

