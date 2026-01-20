"""
姿态检测模块 - 使用MediaPipe Pose（云端兼容版）
"""
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseDetector:
    """身体姿态检测器"""

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        初始化姿态检测器

        Args:
            static_image_mode: 是否为静态图片模式
            model_complexity: 模型复杂度 (0, 1, 2)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.results = None

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.mp_pose = None
            self.mp_draw = None
            self.mp_drawing_styles = None
            self.pose = None

    def detect(self, frame):
        """
        检测图像中的人体姿态

        Args:
            frame: BGR格式的图像

        Returns:
            处理后的图像
        """
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return frame
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 进行检测
        self.results = self.pose.process(rgb_frame)

        return frame

    def draw_landmarks(self, frame, draw_connections=True):
        """
        在图像上绘制姿态关键点

        Args:
            frame: 图像
            draw_connections: 是否绘制连接线

        Returns:
            绘制后的图像
        """
        if not MEDIAPIPE_AVAILABLE:
            return frame
        if self.results and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS if draw_connections else None,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def get_landmarks(self):
        """
        获取检测到的关键点

        Returns:
            关键点列表，如果没有检测到则返回None
        """
        if not MEDIAPIPE_AVAILABLE:
            return None
        if self.results and self.results.pose_landmarks:
            return self.results.pose_landmarks.landmark
        return None

    def get_landmark_by_name(self, name):
        """
        根据名称获取特定关键点

        Args:
            name: 关键点名称 (如 'LEFT_WRIST', 'RIGHT_SHOULDER' 等)

        Returns:
            关键点对象，如果没有找到则返回None
        """
        if not MEDIAPIPE_AVAILABLE:
            return None
        landmarks = self.get_landmarks()
        if landmarks is None:
            return None

        try:
            idx = self.mp_pose.PoseLandmark[name].value
            return landmarks[idx]
        except (KeyError, IndexError):
            return None

    def is_detected(self):
        """检查是否检测到人体"""
        if not MEDIAPIPE_AVAILABLE:
            return False
        return self.results is not None and self.results.pose_landmarks is not None

    def release(self):
        """释放资源"""
        if self.pose:
            self.pose.close()
