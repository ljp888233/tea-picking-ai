"""
手部检测模块 - 使用MediaPipe Hands
云端优化版 - 强制CPU模式
"""
import mediapipe as mp
import cv2


class HandDetector:
    """手部检测器"""
    
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        初始化手部检测器
        
        Args:
            static_image_mode: 是否为静态图片模式
            max_num_hands: 最大检测手数
            model_complexity: 模型复杂度 (0, 1) - 云端默认0避免GPU问题
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.results = None
    
    def detect(self, frame):
        """
        检测图像中的手部
        
        Args:
            frame: BGR格式的图像
            
        Returns:
            处理后的图像
        """
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行检测
        self.results = self.hands.process(rgb_frame)
        
        return frame
    
    def draw_landmarks(self, frame):
        """
        在图像上绘制手部关键点
        
        Args:
            frame: 图像
            
        Returns:
            绘制后的图像
        """
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def get_all_hands(self):
        """
        获取所有检测到的手部信息
        
        Returns:
            手部信息列表，每个元素包含 landmarks 和 handedness
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return []
        
        hands_data = []
        for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
            handedness = "Unknown"
            if self.results.multi_handedness and idx < len(self.results.multi_handedness):
                handedness = self.results.multi_handedness[idx].classification[0].label
            
            hands_data.append({
                'landmarks': hand_landmarks.landmark,
                'handedness': handedness
            })
        
        return hands_data
    
    def get_finger_tips(self):
        """
        获取所有手指尖端的坐标
        
        Returns:
            字典，包含各手指尖端的坐标
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        
        # 只返回第一只手的数据
        landmarks = self.results.multi_hand_landmarks[0].landmark
        
        finger_tips = {
            'thumb': landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
            'index': landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            'middle': landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            'ring': landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            'pinky': landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        }
        
        return finger_tips
    
    def get_pinch_distance(self, hand_landmarks):
        """
        计算拇指和食指之间的距离（捏取动作检测）
        
        Args:
            hand_landmarks: 手部关键点列表
            
        Returns:
            归一化距离值
        """
        thumb_tip = hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # 计算欧氏距离
        distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                   (thumb_tip.y - index_tip.y) ** 2 + 
                   (thumb_tip.z - index_tip.z) ** 2) ** 0.5
        
        return distance
    
    def is_detected(self):
        """检查是否检测到手部"""
        return self.results is not None and self.results.multi_hand_landmarks is not None
    
    def release(self):
        """释放资源"""
        self.hands.close()

