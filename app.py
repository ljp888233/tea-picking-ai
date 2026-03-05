"""
AI采茶动作捕捉系统 - WebRTC云端版
使用 streamlit-webrtc 实现浏览器摄像头访问
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from datetime import datetime
import os

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level

# WebRTC 配置 - STUN/TURN 服务器
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
        {
            "urls": ["turn:a.relay.metered.ca:80", "turn:a.relay.metered.ca:443"],
            "username": "e8dd65b92f6de7b41379c769",
            "credential": "6JKsbXBnUkdfT3Oz"
        },
    ]}
)

# 页面配置
st.set_page_config(page_title="智茶 AI", page_icon="🍵", layout="wide")

# CSS样式
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%); }
    .main-title { text-align: center; font-size: 2.5rem; font-weight: 700; color: #2E7D32; margin-bottom: 1rem; }
    .score-big { font-size: 4rem; font-weight: bold; text-align: center; }
    .score-excellent { color: #2E7D32; }
    .score-good { color: #F9A825; }
    .score-poor { color: #D32F2F; }
    .info-card { background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .feedback-item { padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px; background: #f5f5f5; }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-title">🍵 智茶 AI - 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">实时分析采茶手势，AI智能评分</p>', unsafe_allow_html=True)


class VideoProcessor:
    """视频处理器"""
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.analyzer = TeaPickingAnalyzer()
        self.result = {}
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
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
        
        # 获取统计
        stats = self.analyzer.get_statistics()
        score = stats['current_score']
        
        # 在画面上显示
        color = get_score_color(score)
        cv2.putText(img, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(img, f"Picks: {stats['pick_count']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Hands: {len(hands_data)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 侧边栏
with st.sidebar:
    st.header("🍵 智茶 AI")
    st.divider()
    st.header("📊 使用说明")
    st.markdown("""
    1. 点击 **START** 开启摄像头
    2. 允许浏览器访问摄像头
    3. 将手放在摄像头前
    4. 做出采茶的 **捏取** 动作
    5. 观察实时评分和反馈
    """)
    st.divider()
    st.caption("© 2024 智茶AI")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 实时画面")
    st.info("👆 点击 START 开启摄像头，首次使用请允许浏览器访问摄像头权限")
    
    ctx = webrtc_streamer(
        key="tea-picking",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🏆 实时成绩")
    score_placeholder = st.empty()
    level_placeholder = st.empty()
    
    st.subheader("📈 统计数据")
    stats_placeholder = st.empty()
    
    st.subheader("💡 动作反馈")
    feedback_placeholder = st.empty()

# 更新显示
if ctx.video_processor:
    score = 0
    stats = {'pick_count': 0, 'current_score': 0, 'average_score': 0, 'total_actions': 0}
    feedback = []
    
    if hasattr(ctx.video_processor, 'analyzer'):
        stats = ctx.video_processor.analyzer.get_statistics()
        score = stats['current_score']
    if hasattr(ctx.video_processor, 'result'):
        feedback = ctx.video_processor.result.get('feedback', [])
    
    score_class = "score-excellent" if score >= 70 else ("score-good" if score >= 50 else "score-poor")
    score_placeholder.markdown(f'<div class="score-big {score_class}">{score}</div>', unsafe_allow_html=True)
    level_placeholder.markdown(f'<p style="text-align: center; font-size: 1.5rem;">{get_score_level(score)}</p>', unsafe_allow_html=True)

    stats_placeholder.markdown(f"""
    <div class="info-card">
        <p>🍃 采摘次数: <b>{stats['pick_count']}</b></p>
        <p>📊 平均分: <b>{stats['average_score']}</b></p>
        <p>🎯 总动作数: <b>{stats['total_actions']}</b></p>
    </div>
    """, unsafe_allow_html=True)

    if feedback:
        feedback_html = ""
        for fb in feedback[:5]:
            feedback_html += f'<div class="feedback-item">{fb}</div>'
        feedback_placeholder.markdown(feedback_html, unsafe_allow_html=True)
    else:
        feedback_placeholder.markdown('<div class="feedback-item">○ 等待检测手部动作...</div>', unsafe_allow_html=True)
else:
    # 默认状态
    score_placeholder.markdown('<div class="score-big score-good">--</div>', unsafe_allow_html=True)
    level_placeholder.markdown('<p style="text-align: center; font-size: 1.5rem;">等待开始...</p>', unsafe_allow_html=True)
    stats_placeholder.markdown("""
    <div class="info-card">
        <p>🍃 采摘次数: <b>0</b></p>
        <p>📊 平均分: <b>--</b></p>
        <p>🎯 总动作数: <b>0</b></p>
    </div>
    """, unsafe_allow_html=True)
    feedback_placeholder.markdown('<div class="feedback-item">○ 点击 START 开始检测...</div>', unsafe_allow_html=True)

