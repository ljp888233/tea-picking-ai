"""
AI采茶动作捕捉系统 V2.0 - WebRTC版本
"""
import streamlit as st
import cv2
import numpy as np
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="智茶 AI", page_icon="🍵", layout="wide")

class VideoProcessor:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        self.analyzer = TeaPickingAnalyzer()
        self.score = 0
        self.feedback = []
        self.show_pose = True
        self.show_hands = True
        self.lock = threading.Lock()
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        self.pose_detector.detect(img)
        if self.show_pose:
            self.pose_detector.draw_landmarks(img)
        
        self.hand_detector.detect(img)
        if self.show_hands:
            self.hand_detector.draw_landmarks(img)
        
        hands_data = self.hand_detector.get_all_hands()
        if hands_data:
            result = self.analyzer.analyze_hand(
                hands_data[0]['landmarks'],
                hands_data[0]['handedness']
            )
            with self.lock:
                self.score = result['score']
                self.feedback = result['feedback']
        
        score_color = get_score_color(self.score)
        cv2.putText(img, f"Score: {self.score}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_score(self):
        with self.lock:
            return self.score
    
    def get_feedback(self):
        with self.lock:
            return self.feedback.copy()
    
    def get_stats(self):
        return self.analyzer.get_statistics()


def main():
    st.markdown('<h1 style="text-align:center;color:#2E7D32;">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#666;">🌿 传承千年茶艺，智能科技赋能</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🍵 智茶 AI")
        st.divider()
        show_pose = st.checkbox("显示身体骨骼", value=True)
        show_hands = st.checkbox("显示手部骨骼", value=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 实时画面")
        st.info("👆 点击 START 开启摄像头，首次使用请允许浏览器访问摄像头")
        
        webrtc_streamer(
            key="tea-picking",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("🏆 使用说明")
        st.markdown("""
        1. 点击 **START** 按钮
        2. 允许浏览器访问摄像头
        3. 对准摄像头做采茶动作
        4. 查看实时评分和反馈
        """)


if __name__ == "__main__":
    main()
