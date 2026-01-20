"""
手部检测模块 - 使用MediaPipe Hands（云端兼容版）
"""
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


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
            model_complexity: 模型复杂度 (0, 1)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.results = None

        if MEDIAPIPE_AVAILABLE:
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
        else:
            self.mp_hands = None
            self.mp_draw = None
            self.mp_drawing_styles = None
            self.hands = None

    def detect(self, frame):
        """
        检测图像中的手部

        Args:
            frame: BGR格式的图像

        Returns:
            处理后的图像
        """
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        if not MEDIAPIPE_AVAILABLE:
            return frame
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
        获取所有检测到的手部数据

        Returns:
            列表，每个元素包含 (landmarks, handedness)
        """
        if not MEDIAPIPE_AVAILABLE:
            return []
        hands_data = []
        if self.results and self.results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                handedness = None
                if self.results.multi_handedness:
                    handedness = self.results.multi_handedness[idx].classification[0].label
                hands_data.append({
                    'landmarks': hand_landmarks.landmark,
                    'handedness': handedness  # 'Left' or 'Right'
                })
        return hands_data

    def get_finger_tips(self, hand_landmarks):
        """
        获取指尖关键点

        Args:
            hand_landmarks: 手部关键点列表

        Returns:
            字典，包含各指尖坐标
        """
        if not MEDIAPIPE_AVAILABLE:
            return {}
        tips = {
            'thumb': hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
            'index': hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            'middle': hand_landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            'ring': hand_landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            'pinky': hand_landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        }
        return tips

    def get_pinch_distance(self, hand_landmarks):
        """
        计算拇指和食指的捏取距离

        Args:
            hand_landmarks: 手部关键点列表

        Returns:
            捏取距离（归一化值）
        """
        if not MEDIAPIPE_AVAILABLE:
            return 0.0
        thumb_tip = hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        distance = ((thumb_tip.x - index_tip.x)**2 +
                   (thumb_tip.y - index_tip.y)**2 +
                   (thumb_tip.z - index_tip.z)**2) ** 0.5
        return distance

    def is_detected(self):
        """检查是否检测到手部"""
        if not MEDIAPIPE_AVAILABLE:
            return False
        return self.results is not None and self.results.multi_hand_landmarks is not None

    def get_hand_count(self):
        """获取检测到的手部数量"""
        if not MEDIAPIPE_AVAILABLE:
            return 0
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0

    def release(self):
        """释放资源"""
        if self.hands:
            self.hands.close()"""
手部检测模块 - 使用MediaPipe Hands
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
            model_complexity: 模型复杂度 (0, 1)
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        获取所有检测到的手部数据
        
        Returns:
            列表，每个元素包含 (landmarks, handedness)
        """
        hands_data = []
        if self.results and self.results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                handedness = None
                if self.results.multi_handedness:
                    handedness = self.results.multi_handedness[idx].classification[0].label
                hands_data.append({
                    'landmarks': hand_landmarks.landmark,
                    'handedness': handedness  # 'Left' or 'Right'
                })
        return hands_data
    
    def get_finger_tips(self, hand_landmarks):
        """
        获取指尖关键点
        
        Args:
            hand_landmarks: 手部关键点列表
            
        Returns:
            字典，包含各指尖坐标
        """
        tips = {
            'thumb': hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
            'index': hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            'middle': hand_landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            'ring': hand_landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            'pinky': hand_landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        }
        return tips
    
    def get_pinch_distance(self, hand_landmarks):
        """
        计算拇指和食指的捏取距离
        
        Args:
            hand_landmarks: 手部关键点列表
            
        Returns:
            捏取距离（归一化值）
        """
        thumb_tip = hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        distance = ((thumb_tip.x - index_tip.x)**2 + 
                   (thumb_tip.y - index_tip.y)**2 + 
                   (thumb_tip.z - index_tip.z)**2) ** 0.5
        return distance
    
    def is_detected(self):
        """检查是否检测到手部"""
        return self.results is not None and self.results.multi_hand_landmarks is not None
    
    def get_hand_count(self):
        """获取检测到的手部数量"""
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    
    def release(self):
        """释放资源"""
        self.hands.close()

