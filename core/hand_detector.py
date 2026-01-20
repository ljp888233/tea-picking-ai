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
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        return frame

    def draw_landmarks(self, frame):
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
                    'handedness': handedness
                })
        return hands_data

    def get_finger_tips(self, hand_landmarks):
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
        if not MEDIAPIPE_AVAILABLE:
            return 0.0
        thumb_tip = hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = ((thumb_tip.x - index_tip.x)**2 +
                   (thumb_tip.y - index_tip.y)**2 +
                   (thumb_tip.z - index_tip.z)**2) ** 0.5
        return distance

    def is_detected(self):
        if not MEDIAPIPE_AVAILABLE:
            return False
        return self.results is not None and self.results.multi_hand_landmarks is not None

    def get_hand_count(self):
        if not MEDIAPIPE_AVAILABLE:
            return 0
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0

    def release(self):
        if self.hands:
            self.hands.close()
