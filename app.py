"""
AI采茶动作捕捉系统 V2.0 - 云端简化版
使用 Streamlit 原生摄像头拍照功能
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level

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
    .stApp { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%); }
    .main-title {
        text-align: center; font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(120deg, #2E7D32 0%, #00695C 50%, #1B5E20 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
    }
    .sub-title { text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .score-display { font-size: 5rem; font-weight: 800; text-align: center; text-shadow: 0 0 20px currentColor; }
    .big-number {
        font-size: 3.5rem; font-weight: 700; text-align: center;
        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .feedback-item {
        padding: 0.8rem 1rem; margin: 0.4rem 0; border-radius: 10px;
        background: linear-gradient(135deg, #FAFAFA 0%, #F5F5F5 100%);
        border-left: 4px solid #4CAF50; font-size: 0.95rem;
    }
    .mode-title {
        font-size: 1.1rem; color: #37474F; padding: 0.8rem 1.2rem;
        border-radius: 10px; background: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 100%);
        border-left: 4px solid #00897B; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def rgb_to_hex(bgr_color):
    """BGR颜色转十六进制"""
    return f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"


def main():
    """主函数"""
    # 标题
    st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">☁️ 云端版 | 传承千年茶艺，智能科技赋能</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888; font-size:0.9rem; margin-top:-0.5rem;">Designed by 川农物联网202306李逍遥</p>', unsafe_allow_html=True)

    # 侧边栏
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
        st.subheader("📖 使用说明")
        st.markdown("""
        1. 点击 **Take Photo** 拍照
        2. 允许浏览器访问摄像头
        3. 调整手部姿势后拍照
        4. 查看AI智能评分
        5. 点击 **Clear Photo** 重拍
        """)
        st.divider()
        st.caption("© 2026 智茶AI · Cloud")

    # 初始化检测器
    if 'pose_detector' not in st.session_state:
        st.session_state.pose_detector = PoseDetector()
        st.session_state.hand_detector = HandDetector()
        st.session_state.analyzer = TeaPickingAnalyzer()

    # 根据模式显示界面
    if mode == "🎮 体验模式":
        render_experience_mode()
    elif mode == "📊 效率模式":
        render_efficiency_mode()
    elif mode == "✅ 质控模式":
        render_quality_mode()
    elif mode == "📚 教学模式":
        render_teaching_mode()


def process_image(img):
    """处理图像并返回结果"""
    pose_detector = st.session_state.pose_detector
    hand_detector = st.session_state.hand_detector
    analyzer = st.session_state.analyzer
    
    # 检测
    pose_detector.detect(img)
    pose_detector.draw_landmarks(img)
    
    hand_detector.detect(img)
    hand_detector.draw_landmarks(img)
    
    # 分析
    hands_data = hand_detector.get_all_hands()
    result = {'score': 0, 'feedback': ['未检测到手部'], 'is_pinching': False}
    
    if hands_data:
        result = analyzer.analyze_hand(
            hands_data[0]['landmarks'],
            hands_data[0]['handedness']
        )
    
    stats = analyzer.get_statistics()
    
    return img, result, stats


def render_experience_mode():
    """🎮 体验模式"""
    st.markdown('<p class="mode-title">🎮 体验模式 - 趣味互动，挑战采茶大师！</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 拍照检测")
        img_file = st.camera_input("请将手放在摄像头前，做出采茶姿势后拍照")
        
        if img_file is not None:
            # 读取图像
            image = Image.open(img_file)
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 处理
            processed_img, result, stats = process_image(img)
            
            # 显示
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(processed_img_rgb, caption="检测结果", use_container_width=True)
            
            with col2:
                st.subheader("🏆 你的成绩")
                score = result['score']
                score_color = get_score_color(score)
                st.markdown(
                    f'<p class="score-display" style="color:{rgb_to_hex(score_color)}">{score}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="text-align:center;font-size:1.5rem;">{get_score_level(score)}</p>',
                    unsafe_allow_html=True
                )
                
                st.divider()
                st.subheader("💡 动作反馈")
                for fb in result['feedback']:
                    st.markdown(f'<div class="feedback-item">{fb}</div>', unsafe_allow_html=True)
                
                st.divider()
                st.subheader("📊 统计数据")
                st.markdown(f"""
                - 🍃 采摘次数: **{stats['pick_count']}**
                - 📊 当前评分: **{stats['current_score']}**
                - 📈 平均评分: **{stats['average_score']}**
                """)
        else:
            with col2:
                st.subheader("🏆 你的成绩")
                st.markdown('<p class="score-display" style="color:#4CAF50">--</p>', unsafe_allow_html=True)
                st.markdown('<p style="text-align:center;font-size:1.5rem;">等待拍照...</p>', unsafe_allow_html=True)


def render_efficiency_mode():
    """📊 效率模式"""
    render_experience_mode()  # 简化版用同样界面


def render_quality_mode():
    """✅ 质控模式"""
    render_experience_mode()


def render_teaching_mode():
    """📚 教学模式"""
    st.markdown('<p class="mode-title">📚 教学模式 - 学习标准采茶技艺！</p>', unsafe_allow_html=True)
    
    st.markdown("### 📖 采茶标准动作要领")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🖐️ **步骤1: 手型准备**\n\n拇指与食指自然张开")
    with col2:
        st.info("🌱 **步骤2: 捏取茶芽**\n\n拇指食指轻捏茶芽")
    with col3:
        st.info("🍃 **步骤3: 提拉采摘**\n\n轻轻向上提拉")
    
    st.divider()
    render_experience_mode()


if __name__ == "__main__":
    main()

