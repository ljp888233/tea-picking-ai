"""
AI采茶动作捕捉系统 V2.0 - 云端版
主程序 - Streamlit界面（科技感+茶文化风格）
使用 WebRTC 实现云端摄像头访问
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level, draw_chinese_text

# WebRTC配置
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# 页面配置
st.set_page_config(
    page_title="智茶 AI - 采茶动作捕捉系统",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 科技感+茶文化风格CSS (与原版一致)
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%); }
    .main-title {
        text-align: center; font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(120deg, #2E7D32 0%, #00695C 50%, #1B5E20 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
    }
    .sub-title { text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .tech-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(46,125,50,0.1); margin: 0.5rem 0;
    }
    .score-display {
        font-size: 5rem; font-weight: 800; text-align: center;
        text-shadow: 0 0 20px currentColor;
    }
    .big-number {
        font-size: 3.5rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .achievement-badge {
        display: inline-block; padding: 0.4rem 1rem; margin: 0.3rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-size: 0.85rem; font-weight: 500;
    }
    .feedback-item {
        padding: 0.8rem 1rem; margin: 0.4rem 0; border-radius: 10px;
        background: linear-gradient(135deg, #FAFAFA 0%, #F5F5F5 100%);
        border-left: 4px solid #4CAF50; font-size: 0.95rem;
    }
    .feedback-item.warning { border-left-color: #FF9800; background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%); }
    .feedback-item.error { border-left-color: #F44336; background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); }
    .mode-title {
        font-size: 1.1rem; color: #37474F; padding: 0.8rem 1.2rem;
        border-radius: 10px; background: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 100%);
        border-left: 4px solid #00897B; margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem 1.2rem; border-radius: 12px;
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 4px solid #FF9800; margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem 1.2rem; border-radius: 12px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 4px solid #4CAF50; margin: 0.5rem 0;
    }
    .teaching-step {
        padding: 1.2rem; margin: 0.5rem 0; border-radius: 12px;
        background: linear-gradient(145deg, #E3F2FD 0%, #BBDEFB 100%);
        box-shadow: 0 3px 12px rgba(33,150,243,0.15);
    }
    .stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


class TeaPickingVideoProcessor:
    """WebRTC视频处理器"""
    
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        self.analyzer = TeaPickingAnalyzer()
        self.show_pose = True
        self.show_hands = True
        self.result = {'score': 0, 'feedback': [], 'is_pinching': False}
        self.stats = {'pick_count': 0, 'current_score': 0, 'average_score': 0, 'total_actions': 0}
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # 姿态检测
        self.pose_detector.detect(img)
        if self.show_pose:
            self.pose_detector.draw_landmarks(img)
        
        # 手部检测
        self.hand_detector.detect(img)
        if self.show_hands:
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
        
        # 在画面上显示
        color = get_score_color(score)
        cv2.putText(img, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(img, f"Picks: {self.stats['pick_count']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if hands_data:
            cv2.putText(img, "Hand OK", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def rgb_to_hex(bgr_color):
    """BGR颜色转十六进制"""
    return f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"


def main():
    """主函数"""
    # 标题区域
    st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">☁️ 云端版 | 传承千年茶艺，智能科技赋能</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888; font-size:0.9rem; margin-top:-0.5rem;">Designed by 川农物联网202306李逍遥</p>', unsafe_allow_html=True)

    # 侧边栏设置
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🍵</span>
            <h2 style="color: #2E7D32; margin: 0.5rem 0;">智茶 AI</h2>
            <p style="color: #666; font-size: 0.85rem;">Cloud Edition</p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.subheader("🎯 模式选择")
        mode = st.selectbox(
            "选择体验模式",
            ["🎮 体验模式", "📊 效率模式", "✅ 质控模式", "📚 教学模式"],
            index=0,
            label_visibility="collapsed"
        )

        st.divider()
        st.subheader("👁️ 显示选项")
        show_pose = st.checkbox("显示身体骨骼", value=True)
        show_hands = st.checkbox("显示手部骨骼", value=True)

        st.divider()
        st.subheader("📖 使用说明")
        st.markdown("""
        1. 点击 **START** 开启摄像头
        2. 允许浏览器访问摄像头
        3. 将手放在摄像头前
        4. 做出采茶**捏取**动作
        5. 观察实时评分反馈
        """)
        st.divider()
        st.caption("© 2026 智茶AI · Cloud")

    # 根据模式显示不同界面
    if mode == "🎮 体验模式":
        render_experience_mode(show_pose, show_hands)
    elif mode == "📊 效率模式":
        render_efficiency_mode(show_pose, show_hands)
    elif mode == "✅ 质控模式":
        render_quality_mode(show_pose, show_hands)
    elif mode == "📚 教学模式":
        render_teaching_mode(show_pose, show_hands)


def render_experience_mode(show_pose, show_hands):
    """🎮 体验模式"""
    st.markdown('<p class="mode-title">🎮 体验模式 - 趣味互动，挑战采茶大师！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时画面")
        ctx = webrtc_streamer(
            key="experience",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=TeaPickingVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("🏆 你的成绩")
        st.markdown('<p class="score-display" style="color:#4CAF50">--</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;font-size:1.5rem;">等待开始...</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("🎖️ 成就徽章")
        st.markdown('<span style="color:#999;">点击START开始挑战！</span>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 挑战统计")
        st.markdown("""
        - 🍃 采摘次数: **--**
        - 📊 当前评分: **--**
        - 📈 平均评分: **--**
        """)


def render_efficiency_mode(show_pose, show_hands):
    """📊 效率模式"""
    st.markdown('<p class="mode-title">📊 效率模式 - 统计采摘效率，提升工作表现！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时监控")
        ctx = webrtc_streamer(
            key="efficiency",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=TeaPickingVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("⏱️ 效率数据")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**采摘次数**")
            st.markdown('<p class="big-number">--</p>', unsafe_allow_html=True)
        with col_b:
            st.markdown("**每分钟速度**")
            st.markdown('<p class="big-number">--</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📈 效率趋势")
        st.progress(0, text="目标: 100次")

        st.divider()
        st.subheader("📋 详细统计")
        st.markdown("""
        - ⏱️ 已用时间: **--**
        - 🎯 采摘次数: **--**
        - 📈 平均速度: **--**
        - 💯 平均质量: **--**
        """)


def render_quality_mode(show_pose, show_hands):
    """✅ 质控模式"""
    st.markdown('<p class="mode-title">✅ 质控模式 - 规范动作，保证茶叶品质！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 动作监控")
        ctx = webrtc_streamer(
            key="quality",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=TeaPickingVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("📋 质量评估")
        st.markdown('<p style="font-size:2rem;text-align:center;color:#ff9800">等待检测...</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("⚠️ 实时提醒")
        st.markdown('<div class="success-box">✅ 点击START开始质检</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("✅ 规范检查项")
        st.markdown("""
        - ⬜ 捏取姿势规范
        - ⬜ 手指姿态自然
        - ⬜ 动作稳定流畅
        """)

        st.divider()
        st.subheader("📊 质量统计")
        st.markdown("""
        - 📊 合格率: **--%**
        - 🔢 检测次数: **--**
        - 📈 平均得分: **--**
        """)


def render_teaching_mode(show_pose, show_hands):
    """📚 教学模式"""
    st.markdown('<p class="mode-title">📚 教学模式 - 学习标准采茶技艺！</p>', unsafe_allow_html=True)

    # 教学步骤
    st.markdown("### 📖 采茶标准动作要领")
    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1:
        st.markdown("""
        <div class="teaching-step">
            <h4>🖐️ 步骤1: 手型准备</h4>
            <p>拇指与食指自然张开，其余三指微曲放松</p>
        </div>
        """, unsafe_allow_html=True)
    with step_col2:
        st.markdown("""
        <div class="teaching-step">
            <h4>🌱 步骤2: 捏取茶芽</h4>
            <p>拇指食指轻捏茶芽，力度适中不伤叶片</p>
        </div>
        """, unsafe_allow_html=True)
    with step_col3:
        st.markdown("""
        <div class="teaching-step">
            <h4>🍃 步骤3: 提拉采摘</h4>
            <p>轻轻向上提拉，一芽一叶，动作流畅</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 练习画面")
        ctx = webrtc_streamer(
            key="teaching",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=TeaPickingVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("📝 动作评价")
        st.markdown('<p style="font-size:2rem;text-align:center;color:#4CAF50">-- 分</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 改进建议")
        st.markdown('<div class="feedback-item">○ 点击START开始练习...</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📈 学习进度")
        st.progress(0, text="掌握程度: 0%")


if __name__ == "__main__":
    main()

