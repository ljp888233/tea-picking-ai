"""
采茶动作分析模块
分析采茶动作的规范性并给出评分
"""
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import calculate_angle, calculate_distance, smooth_value


class TeaPickingAnalyzer:
    """采茶动作分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 动作状态
        self.current_state = "待机"
        self.pick_count = 0
        self.last_pinch_distance = None
        self.is_picking = False
        
        # 评分相关
        self.scores_history = []
        self.current_score = 0
        
        # 阈值配置
        self.pinch_threshold = 0.05  # 捏取判定阈值
        self.release_threshold = 0.08  # 释放判定阈值
        
        # 平滑参数
        self.smooth_alpha = 0.3
        
    def analyze_hand(self, hand_landmarks, handedness="Right"):
        """
        分析单只手的采茶动作
        
        Args:
            hand_landmarks: 手部关键点列表
            handedness: 左手/右手
            
        Returns:
            分析结果字典
        """
        result = {
            'pinch_distance': 0,
            'is_pinching': False,
            'hand_angle': 0,
            'score': 0,
            'feedback': []
        }
        
        if hand_landmarks is None:
            return result
        
        # 1. 计算捏取距离（拇指-食指）
        thumb_tip = hand_landmarks[4]   # THUMB_TIP
        index_tip = hand_landmarks[8]   # INDEX_FINGER_TIP
        
        pinch_distance = calculate_distance(thumb_tip, index_tip)
        pinch_distance = smooth_value(pinch_distance, self.last_pinch_distance, self.smooth_alpha)
        self.last_pinch_distance = pinch_distance
        
        result['pinch_distance'] = pinch_distance
        
        # 2. 判断是否在捏取
        if pinch_distance < self.pinch_threshold:
            result['is_pinching'] = True
            if not self.is_picking:
                self.is_picking = True
                self.pick_count += 1
        elif pinch_distance > self.release_threshold:
            result['is_pinching'] = False
            self.is_picking = False
        
        # 3. 计算手腕角度
        wrist = hand_landmarks[0]       # WRIST
        middle_mcp = hand_landmarks[9]  # MIDDLE_FINGER_MCP
        middle_tip = hand_landmarks[12] # MIDDLE_FINGER_TIP
        
        hand_angle = calculate_angle(wrist, middle_mcp, middle_tip)
        result['hand_angle'] = hand_angle
        
        # 4. 评分计算
        score, feedback = self._calculate_score(result, hand_landmarks)
        result['score'] = score
        result['feedback'] = feedback
        
        return result
    
    def _calculate_score(self, analysis_result, hand_landmarks):
        """
        计算采茶动作评分
        
        Returns:
            (score, feedback_list)
        """
        score = 100
        feedback = []
        
        # 评分项1: 捏取姿势 (40分)
        pinch_distance = analysis_result['pinch_distance']
        if analysis_result['is_pinching']:
            # 捏取时，距离越小越好
            pinch_score = max(0, 40 - pinch_distance * 400)
            if pinch_score >= 35:
                feedback.append("✓ 捏取姿势标准")
            elif pinch_score >= 25:
                feedback.append("△ 捏取可以更紧一些")
            else:
                feedback.append("✗ 捏取姿势需要调整")
        else:
            pinch_score = 20  # 未捏取时给基础分
            feedback.append("○ 等待采摘动作...")
        
        # 评分项2: 手指伸展 (30分)
        # 检查其他手指是否自然弯曲（不要太僵硬）
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]
        wrist = hand_landmarks[0]
        
        # 计算其他手指到手腕的距离
        other_fingers_dist = (
            calculate_distance(middle_tip, wrist) +
            calculate_distance(ring_tip, wrist) +
            calculate_distance(pinky_tip, wrist)
        ) / 3
        
        if 0.15 < other_fingers_dist < 0.35:
            finger_score = 30
            feedback.append("✓ 手指姿态自然")
        elif 0.1 < other_fingers_dist < 0.4:
            finger_score = 20
            feedback.append("△ 手指可以更放松")
        else:
            finger_score = 10
            feedback.append("✗ 手指姿态需调整")
        
        # 评分项3: 手部稳定性 (30分)
        # 简化处理：基于手腕位置的稳定性
        stability_score = 25  # 基础分，后续可以加入历史数据对比
        feedback.append("✓ 动作较为稳定")
        
        # 总分
        score = pinch_score + finger_score + stability_score
        score = max(0, min(100, score))
        
        self.current_score = smooth_value(score, self.current_score, 0.2)
        self.scores_history.append(self.current_score)
        
        # 保持历史记录在合理范围
        if len(self.scores_history) > 100:
            self.scores_history = self.scores_history[-100:]
        
        return int(self.current_score), feedback
    
    def analyze_pose(self, pose_landmarks):
        """
        分析身体姿态
        
        Args:
            pose_landmarks: 身体姿态关键点
            
        Returns:
            姿态分析结果
        """
        result = {
            'posture_score': 0,
            'arm_angle': 0,
            'feedback': []
        }
        
        if pose_landmarks is None:
            return result
        
        # 分析手臂角度
        # 右臂: 肩膀-肘-手腕
        right_shoulder = pose_landmarks[12]
        right_elbow = pose_landmarks[14]
        right_wrist = pose_landmarks[16]
        
        arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        result['arm_angle'] = arm_angle
        
        # 评分
        if 60 < arm_angle < 150:
            result['posture_score'] = 90
            result['feedback'].append("✓ 手臂姿势良好")
        elif 45 < arm_angle < 165:
            result['posture_score'] = 70
            result['feedback'].append("△ 手臂可以调整角度")
        else:
            result['posture_score'] = 50
            result['feedback'].append("✗ 手臂角度不太合适")
        
        return result
    
    def get_state_text(self):
        """获取当前状态文字"""
        if self.is_picking:
            return "采摘中 🍃"
        else:
            return "准备中 ⏳"
    
    def get_statistics(self):
        """获取统计数据"""
        avg_score = np.mean(self.scores_history) if self.scores_history else 0
        return {
            'pick_count': self.pick_count,
            'current_score': int(self.current_score),
            'average_score': int(avg_score),
            'total_actions': len(self.scores_history)
        }
    
    def reset(self):
        """重置分析器状态"""
        self.current_state = "待机"
        self.pick_count = 0
        self.last_pinch_distance = None
        self.is_picking = False
        self.scores_history = []
        self.current_score = 0

