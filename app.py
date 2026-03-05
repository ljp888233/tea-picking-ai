"""
AI采茶动作捕捉系统 - 云端WebRTC版
使用 streamlit-webrtc 实现浏览器摄像头访问
"""
import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level

# WebRTC TURN服务器配置
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# 页面配置
st.set_page_config(page_title="智茶 AI - 云端版", page_icon="🍵", layout="wide")

# 复用原版CSS样式
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%); }
    .main-title { text-align: center; font-size: 2.5rem; font-weight: 700; color: #2E7D32; margin-bottom: 0.5rem; }
    .sub-title { text-align: center; color: #666; font-size: 1rem; margin-bottom: 1rem; }
    .score-display { font-size: 4rem; font-weight: 800; text-align: center; }
    .tech-card { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 0.5rem 0; }
    .feedback-item { padding: 0.6rem; margin: 0.3rem 0; border-radius: 8px; background: #f5f5f5; border-left: 3px solid #4CAF50; }
    .stat-value { font-size: 2rem; font-weight: 700; color: #2E7D32; text-align: center; }
    .stat-label { font-size: 0.85rem; color: #666; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">☁️ 云端版 - 实时分析采茶手势，AI智能评分</p>', unsafe_allow_html=True)


class TeaPickingProcessor:
    """视频处理器 - 处理每一帧"""
    
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        self.analyzer = TeaPickingAnalyzer()
        self.result = {'score': 0, 'feedback': [], 'is_pinching': False}
        self.stats = {'pick_count': 0, 'current_score': 0, 'average_score': 0, 'total_actions': 0}
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 镜像
        
        # 姿态检测
        self.pose_detector.detect(img)
        self.pose_detector.draw_landmarks(img)
        
        # 手部检测
        self.hand_detector.detect(img)
        self.hand_detector.draw_landmarks(img)
        
        # 分析动作
        hands_data = self.hand_detector.get_all_hands()
        if hands_data:
            self.result = self.analyzer.analyze_hand(
                hands_data[0]['landmarks'],
                hands_data[0]['handedness']
            )
        
        # 更新统计
        self.stats = self.analyzer.get_statistics()
        score = self.stats['current_score']
        
        # 在画面上显示信息
        color = get_score_color(score)
        cv2.putText(img, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(img, f"Picks: {self.stats['pick_count']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        if hands_data:
            cv2.putText(img, "Hand Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 侧边栏
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 3rem;">??</span>
        <h2 style="color: #2E7D32; margin: 0.5rem 0;">智茶 AI</h2>
        <p style="color: #666; font-size: 0.85rem;">Cloud Edition</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("?? 使用说明")
    st.markdown("""
    1. 点击 **START** 开启摄像头
    2. 允许浏览器访问摄像头权限
    3. 将手放在摄像头前
    4. 做出采茶的**捏取**动作
    5. 观察实时评分和反馈
    """)
    st.divider()
    st.caption("© 2026 智茶AI · Cloud Version")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 实时画面")
    ctx = webrtc_streamer(
        key="tea-picking-cloud",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=TeaPickingProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🏆 实时成绩")
    score_placeholder = st.empty()
    level_placeholder = st.empty()
    
    st.divider()
    st.subheader("📊 统计数据")
    stats_placeholder = st.empty()
    
    st.divider()
    st.subheader("💡 动作反馈")
    feedback_placeholder = st.empty()

# 显示默认状态
score_placeholder.markdown('<p class="stat-value">--</p>', unsafe_allow_html=True)
level_placeholder.markdown('<p style="text-align:center;">等待开始...</p>', unsafe_allow_html=True)
stats_placeholder.markdown("""
<div class="tech-card">
    <p>?? 采摘次数: <b>0</b></p>
    <p>📊 平均分: <b>--</b></p>
    <p>🎯 总动作: <b>0</b></p>
</div>
""", unsafe_allow_html=True)
feedback_placeholder.markdown('<div class="feedback-item">○ 点击 START 开始检测...</div>', unsafe_allow_html=True)
