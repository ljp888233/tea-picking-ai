"""
AI采茶动作捕捉系统 V2.0 - WebRTC云端版
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
from datetime import datetime

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="智茶 AI", page_icon="🍵", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%); }
    .main-title { text-align: center; font-size: 2.5rem; font-weight: 700; color: #2E7D32; }
    .sub-title { text-align: center; color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .score-display { font-size: 4rem; font-weight: 800; text-align: center; }
    .big-number { font-size: 3rem; font-weight: 700; text-align: center; color: #1976D2; }
    .mode-title { font-size: 1.1rem; color: #37474F; padding: 0.8rem; border-radius: 10px; background: #E0F2F1; border-left: 4px solid #00897B; margin-bottom: 1rem; }
    .achievement-badge { display: inline-block; padding: 0.4rem 1rem; margin: 0.3rem; border-radius: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 0.85rem; }
    .feedback-item { padding: 0.8rem; margin: 0.4rem 0; border-radius: 10px; background: #FAFAFA; border-left: 4px solid #4CAF50; }
    .warning-box { padding: 1rem; border-radius: 12px; background: #FFF3E0; border-left: 4px solid #FF9800; }
    .success-box { padding: 1rem; border-radius: 12px; background: #E8F5E9; border-left: 4px solid #4CAF50; }
    .teaching-step { padding: 1rem; margin: 0.5rem 0; border-radius: 12px; background: #E3F2FD; }
</style>
""", unsafe_allow_html=True)


class VideoProcessor:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        self.analyzer = TeaPickingAnalyzer()
        self.lock = threading.Lock()
        self.score = 0
        self.feedback = []
        self.start_time = time.time()
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.pose_detector.detect(img)
        self.pose_detector.draw_landmarks(img)
        self.hand_detector.detect(img)
        self.hand_detector.draw_landmarks(img)
        
        hands_data = self.hand_detector.get_all_hands()
        if hands_data:
            result = self.analyzer.analyze_hand(hands_data[0]['landmarks'], hands_data[0]['handedness'])
            with self.lock:
                self.score = result['score']
                self.feedback = result['feedback']
        
        score_color = get_score_color(self.score)
        cv2.putText(img, f"Score: {self.score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_data(self):
        with self.lock:
            return {'score': self.score, 'feedback': self.feedback.copy(), 'stats': self.analyzer.get_statistics(), 'elapsed': time.time() - self.start_time}


def rgb_to_hex(bgr):
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def main():
    st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">🌿 传承千年茶艺，智能科技赋能</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="text-align:center;"><span style="font-size:3rem;">🍵</span><h2 style="color:#2E7D32;">智茶 AI</h2></div>', unsafe_allow_html=True)
        st.divider()
        user_name = st.text_input("👤 姓名", placeholder="请输入您的姓名")
        st.divider()
        mode = st.selectbox("🎯 模式选择", ["🎮 体验模式", "📊 效率模式", "✅ 质控模式", "📚 教学模式"])
        st.divider()
        st.markdown('<p style="text-align:center;color:#999;font-size:0.8rem;">Version 2.0 WebRTC<br>© 2026 智茶AI</p>', unsafe_allow_html=True)

    if mode == "🎮 体验模式":
        render_experience_mode()
    elif mode == "📊 效率模式":
        render_efficiency_mode()
    elif mode == "✅ 质控模式":
        render_quality_mode()
    elif mode == "📚 教学模式":
        render_teaching_mode()


def render_experience_mode():
    st.markdown('<p class="mode-title">🎮 体验模式 - 趣味互动，挑战采茶大师！</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📹 实时画面")
        st.info("👆 点击 START 开启摄像头")
        webrtc_streamer(key="exp", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        st.subheader("🏆 使用说明")
        st.markdown("""
        1. 点击 **START** 按钮
        2. 允许浏览器访问摄像头
        3. 对准摄像头做采茶动作
        4. 查看实时评分
        """)
        st.divider()
        st.subheader("🎖️ 成就系统")
        st.markdown("""
        - 🌱 初次采摘 - 完成首次采摘
        - 🍃 采茶新秀 - 采摘10次
        - 🌿 采茶达人 - 采摘50次
        - ⭐ 高分选手 - 平均分80+
        """)


def render_efficiency_mode():
    st.markdown('<p class="mode-title">📊 效率模式 - 统计采摘效率！</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📹 实时监控")
        st.info("👆 点击 START 开启摄像头")
        webrtc_streamer(key="eff", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        st.subheader("⏱️ 效率指标")
        st.markdown("""
        - **采摘次数**: 实时统计
        - **每分钟速度**: 自动计算
        - **平均质量**: 动作评分
        """)


def render_quality_mode():
    st.markdown('<p class="mode-title">✅ 质控模式 - 规范动作，保证品质！</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📹 动作监控")
        st.info("👆 点击 START 开启摄像头")
        webrtc_streamer(key="qc", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        st.subheader("✅ 规范检查项")
        st.markdown("""
        - ✅ 捏取姿势规范
        - ✅ 手指姿态自然
        - ✅ 动作稳定流畅
        """)


def render_teaching_mode():
    st.markdown('<p class="mode-title">📚 教学模式 - 学习标准采茶技艺！</p>', unsafe_allow_html=True)
    st.markdown("### 📖 采茶标准动作要领")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="teaching-step"><h4>🖐️ 步骤1</h4><p>拇指与食指自然张开</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="teaching-step"><h4>🌱 步骤2</h4><p>拇指食指轻捏茶芽</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="teaching-step"><h4>🍃 步骤3</h4><p>轻轻向上提拉采摘</p></div>', unsafe_allow_html=True)
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📹 练习画面")
        st.info("👆 点击 START 开启摄像头")
        webrtc_streamer(key="teach", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        st.subheader("💡 学习提示")
        st.markdown("""
        1. 观看上方动作要领
        2. 开启摄像头练习
        3. 根据评分调整动作
        4. 反复练习直到熟练
        """)


if __name__ == "__main__":
    main()
