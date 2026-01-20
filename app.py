"""
AI采茶动作捕捉系统 V2.0 - WebRTC云端版
支持四种模式：体验/效率/质控/教学
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
import os

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level

# WebRTC 配置
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 页面配置
st.set_page_config(
    page_title="智茶 AI - 采茶动作捕捉系统",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%);
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2E7D32 0%, #00695C 50%, #1B5E20 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .score-display {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
    }
    .big-number {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #1976D2;
    }
    .mode-title {
        font-size: 1.1rem;
        color: #37474F;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 100%);
        border-left: 4px solid #00897B;
        margin-bottom: 1rem;
    }
    .achievement-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 0.85rem;
    }
    .feedback-item {
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        background: #FAFAFA;
        border-left: 4px solid #4CAF50;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 4px solid #FF9800;
    }
    .success-box {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 4px solid #4CAF50;
    }
    .teaching-step {
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        background: linear-gradient(145deg, #E3F2FD 0%, #BBDEFB 100%);
    }
</style>
""", unsafe_allow_html=True)


class VideoProcessor:
    """视频处理器 - 处理每一帧"""
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        self.analyzer = TeaPickingAnalyzer()
        self.lock = threading.Lock()
        self.score = 0
        self.feedback = []
        self.show_pose = True
        self.show_hands = True
        self.start_time = time.time()
    
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
        
        # 分析手部动作
        hands_data = self.hand_detector.get_all_hands()
        if hands_data:
            result = self.analyzer.analyze_hand(
                hands_data[0]['landmarks'],
                hands_data[0]['handedness']
            )
            with self.lock:
                self.score = result['score']
                self.feedback = result['feedback']
        
        # 在画面上显示分数
        score_color = get_score_color(self.score)
        cv2.putText(img, f"Score: {self.score}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_data(self):
        with self.lock:
            return {
                'score': self.score,
                'feedback': self.feedback.copy(),
                'stats': self.analyzer.get_statistics(),
                'elapsed': time.time() - self.start_time
            }


def rgb_to_hex(bgr_color):
    return f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"


def main():
    st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">🌿 传承千年茶艺，智能科技赋能 | AI-Powered Tea Picking</p>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🍵</span>
            <h2 style="color: #2E7D32; margin: 0.5rem 0;">智茶 AI</h2>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        
        st.subheader("👤 使用者信息")
        user_name = st.text_input("姓名", placeholder="请输入您的姓名")
        
        st.divider()
        st.subheader("🎯 模式选择")
        mode = st.selectbox(
            "选择体验模式",
            ["🎮 体验模式", "📊 效率模式", "✅ 质控模式", "📚 教学模式"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.subheader("👁️ 显示选项")
        show_pose = st.checkbox("显示身体骨骼", value=True)
        show_hands = st.checkbox("显示手部骨骼", value=True)
        
        st.divider()
        st.markdown("""
        <div style="text-align:center; color: #999; font-size: 0.8rem;">
            <p>Version 2.0 WebRTC</p>
            <p>© 2026 智茶AI</p>
        </div>
        """, unsafe_allow_html=True)

    # 根据模式渲染
    if mode == "🎮 体验模式":
        render_experience_mode(show_pose, show_hands, user_name)
    elif mode == "📊 效率模式":
        render_efficiency_mode(show_pose, show_hands, user_name)
    elif mode == "✅ 质控模式":
        render_quality_mode(show_pose, show_hands, user_name)
    elif mode == "📚 教学模式":
        render_teaching_mode(show_pose, show_hands, user_name)
        def render_experience_mode(show_pose, show_hands, user_name):
    """🎮 体验模式"""
    st.markdown('<p class="mode-title">🎮 体验模式 - 趣味互动，挑战采茶大师！</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 实时画面")
        st.info("👆 点击 START 开启摄像头，首次使用请允许浏览器访问摄像头权限")
        
        ctx = webrtc_streamer(
            key="experience",
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
        
        st.divider()
        st.subheader("🎖️ 成就徽章")
        achievement_placeholder = st.empty()
        
        st.divider()
        st.subheader("📊 统计数据")
        stats_placeholder = st.empty()
    
    # 实时更新UI
    if ctx.video_processor:
        while ctx.state.playing:
            data = ctx.video_processor.get_data()
            score = data['score']
            stats = data['stats']
            
            score_color = rgb_to_hex(get_score_color(score))
            score_placeholder.markdown(f'<p class="score-display" style="color:{score_color}">{score}</p>', unsafe_allow_html=True)
            level_placeholder.markdown(f'<p style="text-align:center;font-size:1.5rem;">{get_score_level(score)}</p>', unsafe_allow_html=True)
            
            achievements = []
            if stats['pick_count'] >= 1: achievements.append("🌱 初次采摘")
            if stats['pick_count'] >= 10: achievements.append("🍃 采茶新秀")
            if stats['pick_count'] >= 50: achievements.append("🌿 采茶达人")
            if stats['average_score'] >= 80: achievements.append("⭐ 高分选手")
            
            if achievements:
                achievement_placeholder.markdown("".join([f'<span class="achievement-badge">{a}</span>' for a in achievements]), unsafe_allow_html=True)
            else:
                achievement_placeholder.markdown('<span style="color:#999;">继续努力解锁成就！</span>', unsafe_allow_html=True)
            
            stats_placeholder.markdown(f"""
            - 🍃 采摘次数: **{stats['pick_count']}**
            - 📊 当前评分: **{stats['current_score']}**
            - 📈 平均评分: **{stats['average_score']}**
            """)
            
            time.sleep(0.5)


def render_efficiency_mode(show_pose, show_hands, user_name):
    """📊 效率模式"""
    st.markdown('<p class="mode-title">📊 效率模式 - 统计采摘效率，提升工作表现！</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 实时监控")
        st.info("👆 点击 START 开启摄像头")
        
        ctx = webrtc_streamer(
            key="efficiency",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("⏱️ 效率数据")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**采摘次数**")
            count_placeholder = st.empty()
        with col_b:
            st.markdown("**每分钟速度**")
            speed_placeholder = st.empty()
        
        st.divider()
        st.subheader("📈 效率趋势")
        chart_placeholder = st.empty()
        
        st.divider()
        st.subheader("📋 详细统计")
        detail_placeholder = st.empty()
    
    if ctx.video_processor:
        while ctx.state.playing:
            data = ctx.video_processor.get_data()
            stats = data['stats']
            elapsed = data['elapsed']
            
            count_placeholder.markdown(f'<p class="big-number">{stats["pick_count"]}</p>', unsafe_allow_html=True)
            speed = stats['pick_count'] / (elapsed / 60) if elapsed > 0 else 0
            speed_placeholder.markdown(f'<p class="big-number">{speed:.1f}</p>', unsafe_allow_html=True)
            
            chart_placeholder.progress(min(stats['pick_count'] / 100, 1.0), text=f"目标: 100次")
            
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            detail_placeholder.markdown(f"""
            - ⏱️ 已用时间: **{minutes}分{seconds}秒**
            - 🎯 采摘次数: **{stats['pick_count']}**
            - 📈 平均速度: **{speed:.1f}次/分钟**
            - 💯 平均质量: **{stats['average_score']}分**
            """)
            
            time.sleep(0.5)


def render_quality_mode(show_pose, show_hands, user_name):
    """✅ 质控模式"""
    st.markdown('<p class="mode-title">✅ 质控模式 - 规范动作，保证茶叶品质！</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 动作监控")
        st.info("👆 点击 START 开启摄像头")
        
        ctx = webrtc_streamer(
            key="quality",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📋 质量评估")
        quality_placeholder = st.empty()
        
        st.divider()
        st.subheader("⚠️ 实时提醒")
        warning_placeholder = st.empty()
        
        st.divider()
        st.subheader("✅ 规范检查项")
        checklist_placeholder = st.empty()
        
        st.divider()
        st.subheader("📊 质量统计")
        report_placeholder = st.empty()
    
    if ctx.video_processor:
        while ctx.state.playing:
            data = ctx.video_processor.get_data()
            score = data['score']
            feedback = data['feedback']
            stats = data['stats']
            
            quality_level = "优秀 ✅" if score >= 80 else "良好 👍" if score >= 60 else "需改进 ⚠️"
            quality_color = "#4caf50" if score >= 80 else "#ff9800" if score >= 60 else "#f44336"
            quality_placeholder.markdown(f'<p style="font-size:2rem;text-align:center;color:{quality_color}">{quality_level}</p>', unsafe_allow_html=True)
            
            warnings = [fb for fb in feedback if '✗' in fb or '△' in fb]
            if warnings:
                warning_placeholder.markdown(f'<div class="warning-box">{"<br>".join(warnings)}</div>', unsafe_allow_html=True)
            else:
                warning_placeholder.markdown('<div class="success-box">✅ 动作规范，继续保持！</div>', unsafe_allow_html=True)
            
            checklist_placeholder.markdown(f"""
            - {'✅' if score >= 70 else '❌'} 捏取姿势规范
            - {'✅' if score >= 60 else '❌'} 手指姿态自然
            - {'✅' if score >= 50 else '❌'} 动作稳定流畅
            """)
            
            good_rate = (stats['average_score'] / 100) * 100 if stats['average_score'] > 0 else 0
            report_placeholder.markdown(f"""
            - 📊 合格率: **{good_rate:.1f}%**
            - 🔢 检测次数: **{stats['total_actions']}**
            - 📈 平均得分: **{stats['average_score']}**
            """)
            
            time.sleep(0.5)


def render_teaching_mode(show_pose, show_hands, user_name):
    """📚 教学模式"""
    st.markdown('<p class="mode-title">📚 教学模式 - 学习标准采茶技艺！</p>', unsafe_allow_html=True)
    
    st.markdown("### 📖 采茶标准动作要领")
    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1:
        st.markdown('<div class="teaching-step"><h4>🖐️ 步骤1: 手型准备</h4><p>拇指与食指自然张开，其余三指微曲放松</p></div>', unsafe_allow_html=True)
    with step_col2:
        st.markdown('<div class="teaching-step"><h4>🌱 步骤2: 捏取茶芽</h4><p>拇指食指轻捏茶芽，力度适中不伤叶片</p></div>', unsafe_allow_html=True)
    with step_col3:
        st.markdown('<div class="teaching-step"><h4>🍃 步骤3: 提拉采摘</h4><p>轻轻向上提拉，一芽一叶或一芽两叶</p></div>', unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 练习画面")
        st.info("👆 点击 START 开启摄像头")
        
        ctx = webrtc_streamer(
            key="teaching",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📝 动作评价")
        score_placeholder = st.empty()
        
        st.divider()
        st.subheader("💡 改进建议")
        feedback_placeholder = st.empty()
        
        st.divider()
        st.subheader("📈 学习进度")
        progress_placeholder = st.empty()
    
    if ctx.video_processor:
        while ctx.state.playing:
            data = ctx.video_processor.get_data()
            score = data['score']
            feedback = data['feedback']
            stats = data['stats']
            
            score_color = rgb_to_hex(get_score_color(score))
            grade = "优秀" if score >= 80 else "良好" if score >= 60 else "继续练习"
            score_placeholder.markdown(f'<p style="font-size:2rem;text-align:center;color:{score_color}">{score}分 - {grade}</p>', unsafe_allow_html=True)
            
            if feedback:
                feedback_placeholder.markdown("".join([f'<div class="feedback-item">{fb}</div>' for fb in feedback]), unsafe_allow_html=True)
            else:
                feedback_placeholder.markdown('<div class="feedback-item">请开始练习采茶动作</div>', unsafe_allow_html=True)
            
            progress_pct = min(stats['average_score'] / 100, 1.0)
            progress_placeholder.progress(progress_pct, text=f"掌握程度: {int(progress_pct*100)}%")
            
            time.sleep(0.5)


if __name__ == "__main__":
    main()
