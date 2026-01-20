"""
AI采茶动作捕捉系统 V2.0
主程序 - Streamlit界面（科技感+茶文化风格）
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import random
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from core.pose_detector import PoseDetector
from core.hand_detector import HandDetector
from core.action_analyzer import TeaPickingAnalyzer
from utils.helpers import get_score_color, get_score_level, draw_chinese_text


# 页面配置
st.set_page_config(
    page_title="智茶 AI - 采茶动作捕捉系统",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 科技感+茶文化风格CSS
st.markdown("""
<style>
    /* 全局浅绿色背景 */
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%);
    }

    /* 主标题 - 科技感渐变 */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2E7D32 0%, #00695C 50%, #1B5E20 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* 科技感卡片 */
    .tech-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08), 0 0 40px rgba(46,125,50,0.05);
        border: 1px solid rgba(46,125,50,0.1);
        margin: 0.5rem 0;
    }

    /* 分数显示 - 大号霓虹效果 */
    .score-display {
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        text-shadow: 0 0 20px currentColor;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    /* 大数字显示 */
    .big-number {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* 等级徽章 */
    .level-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(46,125,50,0.3);
        text-align: center;
        width: 100%;
    }

    /* 成就徽章 - 更炫酷 */
    .achievement-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(102,126,234,0.3);
        transition: transform 0.2s ease;
    }

    .achievement-badge:hover {
        transform: scale(1.05);
    }

    /* 状态指示器 */
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border: 1px solid #A5D6A7;
    }

    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #4CAF50;
        animation: blink 1s ease-in-out infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* 反馈项 */
    .feedback-item {
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        background: linear-gradient(135deg, #FAFAFA 0%, #F5F5F5 100%);
        border-left: 4px solid #4CAF50;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    .feedback-item:hover {
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .feedback-item.warning {
        border-left-color: #FF9800;
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
    }

    .feedback-item.error {
        border-left-color: #F44336;
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
    }

    /* 模式标题 */
    .mode-title {
        font-size: 1.1rem;
        color: #37474F;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 100%);
        border-left: 4px solid #00897B;
        margin-bottom: 1rem;
    }

    /* 警告/成功框 */
    .warning-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 4px solid #FF9800;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(255,152,0,0.15);
    }

    .success-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(76,175,80,0.15);
    }

    /* 教学步骤卡片 */
    .teaching-step {
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        background: linear-gradient(145deg, #E3F2FD 0%, #BBDEFB 100%);
        border: none;
        box-shadow: 0 3px 12px rgba(33,150,243,0.15);
        transition: transform 0.2s ease;
    }

    .teaching-step:hover {
        transform: translateY(-3px);
    }

    /* 视频容器 */
    .video-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border: 3px solid rgba(46,125,50,0.2);
    }

    /* 统计数据网格 */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }

    .stat-item {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2E7D32;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }

    /* 茶叶装饰元素 */
    .tea-decoration {
        position: fixed;
        font-size: 3rem;
        opacity: 0.1;
        z-index: -1;
    }

    /* 侧边栏美化 */
    .css-1d391kg {
        background: linear-gradient(180deg, #E8F5E9 0%, #C8E6C9 100%);
    }

    /* 按钮美化 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* 进度条美化 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #43A047 0%, #2E7D32 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""

    # 标题区域
    st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">🌿 传承千年茶艺，智能科技赋能 | AI-Powered Tea Picking Motion Capture</p>', unsafe_allow_html=True)

    # 侧边栏设置
    with st.sidebar:
        # Logo区域
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-size: 3rem;">🍵</span>
            <h2 style="color: #2E7D32; margin: 0.5rem 0;">智茶 AI</h2>
            <p style="color: #666; font-size: 0.85rem;">Tea Picking AI System</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # 用户标识
        st.subheader("👤 使用者信息")
        user_name = st.text_input("姓名", value="", placeholder="请输入您的姓名", key="user_name")
        if not user_name:
            st.caption("⚠️ 请输入姓名以便导出数据时区分")

        st.divider()

        st.subheader("🎯 模式选择")
        mode = st.selectbox(
            "选择体验模式",
            ["🎮 体验模式", "📊 效率模式", "✅ 质控模式", "📚 教学模式"],
            index=0,
            label_visibility="collapsed"
        )

        st.divider()

        # 检测设置
        st.subheader("⚙️ 检测参数")
        detection_confidence = st.slider("检测置信度", 0.3, 1.0, 0.5, 0.1)
        tracking_confidence = st.slider("跟踪置信度", 0.3, 1.0, 0.5, 0.1)

        st.divider()

        # 显示设置
        st.subheader("👁️ 显示选项")
        show_pose = st.checkbox("显示身体骨骼", value=True)
        show_hands = st.checkbox("显示手部骨骼", value=True)
        show_fps = st.checkbox("显示帧率", value=True)

        st.divider()

        # 操作按钮
        if st.button("🔄 重置统计", use_container_width=True):
            reset_session_state()
            st.success("✅ 统计已重置！")

        # 版本信息
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0; color: #999; font-size: 0.8rem;">
            <p>Version 2.0</p>
            <p>© 2026 智茶AI</p>
        </div>
        """, unsafe_allow_html=True)

    # 根据模式显示不同界面
    if mode == "🎮 体验模式":
        render_experience_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps)
    elif mode == "📊 效率模式":
        render_efficiency_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps)
    elif mode == "✅ 质控模式":
        render_quality_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps)
    elif mode == "📚 教学模式":
        render_teaching_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps)


def reset_session_state():
    """重置所有session状态"""
    keys_to_reset = ['analyzer', 'pick_count', 'start_time', 'scores_history', 'running', 'sim_score', 'sim_count']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def render_experience_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps):
    """🎮 体验模式 - 趣味评分、等级称号、成就系统"""

    st.markdown('<p class="mode-title">🎮 体验模式 - 趣味互动，挑战采茶大师！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时画面")
        video_placeholder = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶️ 开始挑战", use_container_width=True, type="primary", key="exp_start")
        with btn_col2:
            stop_btn = st.button("⏹️ 结束挑战", use_container_width=True, key="exp_stop")

    with col2:
        st.subheader("🏆 你的成绩")
        score_placeholder = st.empty()
        level_placeholder = st.empty()

        st.divider()
        st.subheader("🎖️ 成就徽章")
        achievement_placeholder = st.empty()

        st.divider()
        st.subheader("📊 挑战统计")
        stats_placeholder = st.empty()

    run_detection_loop(start_btn, stop_btn, video_placeholder,
                       [score_placeholder, level_placeholder, achievement_placeholder, stats_placeholder],
                       detection_confidence, tracking_confidence, show_pose, show_hands, show_fps,
                       mode="experience")


def render_efficiency_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps):
    """📊 效率模式 - 采摘计数、速度统计、数据分析"""

    st.markdown('<p class="mode-title">📊 效率模式 - 统计采摘效率，提升工作表现！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时监控")
        video_placeholder = st.empty()

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("▶️ 开始计时", use_container_width=True, type="primary", key="eff_start")
        with btn_col2:
            stop_btn = st.button("⏹️ 停止计时", use_container_width=True, key="eff_stop")
        with btn_col3:
            export_btn = st.button("🎴 生成成绩卡", use_container_width=True, key="eff_export")

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

    # 处理导出
    if export_btn:
        export_data("efficiency")

    run_detection_loop(start_btn, stop_btn, video_placeholder,
                       [count_placeholder, speed_placeholder, chart_placeholder, detail_placeholder],
                       detection_confidence, tracking_confidence, show_pose, show_hands, show_fps,
                       mode="efficiency")


def render_quality_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps):
    """✅ 质控模式 - 动作规范检测、实时提醒"""

    st.markdown('<p class="mode-title">✅ 质控模式 - 规范动作，保证茶叶品质！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 动作监控")
        video_placeholder = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶️ 开始质检", use_container_width=True, type="primary", key="qc_start")
        with btn_col2:
            stop_btn = st.button("⏹️ 停止质检", use_container_width=True, key="qc_stop")

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

    run_detection_loop(start_btn, stop_btn, video_placeholder,
                       [quality_placeholder, warning_placeholder, checklist_placeholder, report_placeholder],
                       detection_confidence, tracking_confidence, show_pose, show_hands, show_fps,
                       mode="quality")


def render_teaching_mode(detection_confidence, tracking_confidence, show_pose, show_hands, show_fps):
    """📚 教学模式 - 标准动作演示、对比纠正"""

    st.markdown('<p class="mode-title">📚 教学模式 - 学习标准采茶技艺！</p>', unsafe_allow_html=True)

    # 教学步骤展示 - 更美观的卡片
    st.markdown("### 📖 采茶标准动作要领")

    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1:
        st.markdown("""
        <div class="teaching-step">
            <h4>🖐️ 步骤1: 手型准备</h4>
            <p>拇指与食指自然张开，其余三指微曲放松，保持手部灵活</p>
        </div>
        """, unsafe_allow_html=True)
    with step_col2:
        st.markdown("""
        <div class="teaching-step">
            <h4>🌱 步骤2: 捏取茶芽</h4>
            <p>拇指食指轻捏茶芽，力度适中不伤叶片，精准定位</p>
        </div>
        """, unsafe_allow_html=True)
    with step_col3:
        st.markdown("""
        <div class="teaching-step">
            <h4>🍃 步骤3: 提拉采摘</h4>
            <p>轻轻向上提拉，一芽一叶或一芽两叶，动作流畅</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 练习画面")
        video_placeholder = st.empty()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶️ 开始练习", use_container_width=True, type="primary", key="teach_start")
        with btn_col2:
            stop_btn = st.button("⏹️ 结束练习", use_container_width=True, key="teach_stop")

    with col2:
        st.subheader("📝 动作评价")
        score_placeholder = st.empty()

        st.divider()
        st.subheader("💡 改进建议")
        feedback_placeholder = st.empty()

        st.divider()
        st.subheader("📈 学习进度")
        progress_placeholder = st.empty()

    run_detection_loop(start_btn, stop_btn, video_placeholder,
                       [score_placeholder, feedback_placeholder, progress_placeholder, None],
                       detection_confidence, tracking_confidence, show_pose, show_hands, show_fps,
                       mode="teaching")


def run_detection_loop(start_btn, stop_btn, video_placeholder, placeholders,
                       detection_confidence, tracking_confidence, show_pose, show_hands, show_fps,
                       mode="experience"):
    """通用检测循环"""

    # 初始化session_state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'scores_history' not in st.session_state:
        st.session_state.scores_history = []

    if start_btn:
        st.session_state.running = True
        st.session_state.start_time = time.time()

        try:
            # 初始化检测器
            pose_detector = PoseDetector(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            hand_detector = HandDetector(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            analyzer = TeaPickingAnalyzer()

            # 打开摄像头
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ 无法打开摄像头！")
                st.markdown("""
                **可能的原因：**
                - 摄像头未连接或被占用
                - 摄像头权限未开启
                - 驱动程序问题

                **解决方法：**
                1. 检查摄像头是否正确连接
                2. 关闭其他使用摄像头的程序
                3. 在系统设置中允许浏览器访问摄像头
                """)
                st.session_state.running = False
            else:
                st.success("✅ 摄像头已连接，开始检测...")

                fps_time = time.time()
                frame_count = 0
                fps = 0

                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ 无法读取摄像头画面，请检查连接")
                        break

                    frame = cv2.flip(frame, 1)

                    # 检测
                    pose_detector.detect(frame)
                    if show_pose:
                        pose_detector.draw_landmarks(frame)

                    hand_detector.detect(frame)
                    if show_hands:
                        hand_detector.draw_landmarks(frame)

                    # 分析
                    hands_data = hand_detector.get_all_hands()
                    hand_result = {'score': 0, 'feedback': [], 'is_pinching': False}

                    if hands_data:
                        hand_result = analyzer.analyze_hand(
                            hands_data[0]['landmarks'],
                            hands_data[0]['handedness']
                        )

                    # FPS计算
                    frame_count += 1
                    if frame_count >= 10:
                        fps = frame_count / (time.time() - fps_time)
                        fps_time = time.time()
                        frame_count = 0

                    if show_fps:
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    score = hand_result['score']
                    score_color = get_score_color(score)
                    cv2.putText(frame, f"Score: {score}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # 根据模式更新不同的UI
                    stats = analyzer.get_statistics()
                    elapsed_time = time.time() - st.session_state.start_time

                    # 保存数据到session_state供导出使用
                    st.session_state.stats = stats
                    if score > 0:
                        if 'scores_history' not in st.session_state:
                            st.session_state.scores_history = []
                        if len(st.session_state.scores_history) == 0 or st.session_state.scores_history[-1] != score:
                            st.session_state.scores_history.append(score)

                    update_mode_ui(mode, placeholders, score, hand_result, stats, elapsed_time)

                    time.sleep(0.01)

                cap.release()
                pose_detector.release()
                hand_detector.release()
                st.info("检测已停止")

        except Exception as e:
            st.error(f"❌ 发生错误: {str(e)}")
            st.markdown("""
            **请尝试：**
            1. 刷新页面重试
            2. 检查摄像头连接
            3. 重启程序
            """)
            st.session_state.running = False

    if stop_btn:
        st.session_state.running = False


def update_mode_ui(mode, placeholders, score, hand_result, stats, elapsed_time):
    """根据模式更新UI"""

    if mode == "experience":
        # 体验模式: [score, level, achievement, stats]
        score_color = get_score_color(score)
        placeholders[0].markdown(
            f'<p class="score-display" style="color:{rgb_to_hex(score_color)}">{score}</p>',
            unsafe_allow_html=True
        )
        placeholders[1].markdown(
            f'<p style="text-align:center;font-size:1.5rem;">{get_score_level(score)}</p>',
            unsafe_allow_html=True
        )
        # 成就徽章
        achievements = []
        if stats['pick_count'] >= 1:
            achievements.append("🌱 初次采摘")
        if stats['pick_count'] >= 10:
            achievements.append("🍃 采茶新秀")
        if stats['pick_count'] >= 50:
            achievements.append("🌿 采茶达人")
        if stats['average_score'] >= 80:
            achievements.append("⭐ 高分选手")

        achievement_html = "".join([f'<span class="achievement-badge">{a}</span>' for a in achievements])
        if not achievements:
            achievement_html = '<span style="color:#999;">继续努力解锁成就！</span>'
        placeholders[2].markdown(achievement_html, unsafe_allow_html=True)

        placeholders[3].markdown(f"""
        - 🍃 采摘次数: **{stats['pick_count']}**
        - 📊 当前评分: **{stats['current_score']}**
        - 📈 平均评分: **{stats['average_score']}**
        """)

    elif mode == "efficiency":
        # 效率模式: [count, speed, chart, detail]
        placeholders[0].markdown(f'<p class="big-number">{stats["pick_count"]}</p>', unsafe_allow_html=True)

        speed = stats['pick_count'] / (elapsed_time / 60) if elapsed_time > 0 else 0
        placeholders[1].markdown(f'<p class="big-number">{speed:.1f}</p>', unsafe_allow_html=True)

        # 简单的进度条代替图表
        placeholders[2].progress(min(stats['pick_count'] / 100, 1.0), text=f"目标: 100次")

        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        placeholders[3].markdown(f"""
        - ⏱️ 已用时间: **{minutes}分{seconds}秒**
        - 🎯 采摘次数: **{stats['pick_count']}**
        - 📈 平均速度: **{speed:.1f}次/分钟**
        - 💯 平均质量: **{stats['average_score']}分**
        """)

    elif mode == "quality":
        # 质控模式: [quality, warning, checklist, report]
        quality_level = "优秀 ✅" if score >= 80 else "良好 👍" if score >= 60 else "需改进 ⚠️"
        quality_color = "#4caf50" if score >= 80 else "#ff9800" if score >= 60 else "#f44336"
        placeholders[0].markdown(
            f'<p style="font-size:2rem;text-align:center;color:{quality_color}">{quality_level}</p>',
            unsafe_allow_html=True
        )

        # 警告提示
        warnings = []
        for fb in hand_result['feedback']:
            if '✗' in fb or '△' in fb:
                warnings.append(fb)

        if warnings:
            warning_html = '<div class="warning-box">' + '<br>'.join(warnings) + '</div>'
        else:
            warning_html = '<div class="success-box">✅ 动作规范，继续保持！</div>'
        placeholders[1].markdown(warning_html, unsafe_allow_html=True)

        # 检查项
        checklist = f"""
        - {'✅' if score >= 70 else '❌'} 捏取姿势规范
        - {'✅' if score >= 60 else '❌'} 手指姿态自然
        - {'✅' if score >= 50 else '❌'} 动作稳定流畅
        """
        placeholders[2].markdown(checklist)

        # 质量报告
        good_rate = (stats['average_score'] / 100) * 100 if stats['average_score'] > 0 else 0
        placeholders[3].markdown(f"""
        - 📊 合格率: **{good_rate:.1f}%**
        - 🔢 检测次数: **{stats['total_actions']}**
        - 📈 平均得分: **{stats['average_score']}**
        """)

    elif mode == "teaching":
        # 教学模式: [score, feedback, progress, None]
        score_color = get_score_color(score)
        grade = "优秀" if score >= 80 else "良好" if score >= 60 else "继续练习"
        placeholders[0].markdown(
            f'<p style="font-size:2.5rem;text-align:center;color:{rgb_to_hex(score_color)}">{score}分 - {grade}</p>',
            unsafe_allow_html=True
        )

        # 改进建议
        feedback_html = ""
        for fb in hand_result['feedback']:
            feedback_html += f'<div class="feedback-item">{fb}</div>'
        placeholders[1].markdown(feedback_html, unsafe_allow_html=True)

        # 学习进度
        progress_pct = min(stats['average_score'] / 100, 1.0)
        placeholders[2].progress(progress_pct, text=f"掌握程度: {int(progress_pct*100)}%")


def rgb_to_hex(bgr_color):
    """BGR颜色转十六进制"""
    return f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"


def export_data(mode):
    """生成成绩卡片图片 - 保存到项目data文件夹"""
    from datetime import datetime
    from PIL import Image, ImageDraw, ImageFont
    import os

    # 获取用户名
    user_name = st.session_state.get('user_name', '').strip()
    if not user_name:
        st.warning("⚠️ 请先在侧边栏输入您的姓名！")
        return

    # 获取session中的数据
    stats = st.session_state.get('stats', {})
    scores_history = st.session_state.get('scores_history', [])

    if not stats and not scores_history:
        st.warning("⚠️ 暂无数据可导出，请先开始检测！")
        return

    # 创建data文件夹
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建成绩卡片图片
    width, height = 600, 800

    # 创建渐变背景
    img = Image.new('RGB', (width, height), '#E8F5E9')
    draw = ImageDraw.Draw(img)

    # 绘制渐变背景
    for y in range(height):
        r = int(232 - (y / height) * 30)
        g = int(245 - (y / height) * 20)
        b = int(233 - (y / height) * 30)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 绘制装饰边框
    draw.rectangle([20, 20, width-20, height-20], outline='#2E7D32', width=3)
    draw.rectangle([30, 30, width-30, height-30], outline='#81C784', width=1)

    # 尝试加载字体，如果失败则使用默认字体
    try:
        title_font = ImageFont.truetype("msyh.ttc", 36)
        large_font = ImageFont.truetype("msyh.ttc", 48)
        normal_font = ImageFont.truetype("msyh.ttc", 24)
        small_font = ImageFont.truetype("msyh.ttc", 18)
    except:
        title_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # 绘制标题
    draw.text((width//2, 60), "智茶AI", font=title_font, fill='#1B5E20', anchor='mm')
    draw.text((width//2, 100), "- 采茶成绩卡 -", font=normal_font, fill='#2E7D32', anchor='mm')

    # 分隔线
    draw.line([(50, 140), (width-50, 140)], fill='#81C784', width=2)

    # 用户信息
    draw.text((width//2, 180), f"使用者: {user_name}", font=normal_font, fill='#333333', anchor='mm')
    draw.text((width//2, 220), datetime.now().strftime("%Y年%m月%d日 %H:%M"), font=small_font, fill='#666666', anchor='mm')

    # 核心数据区域
    score = stats.get('current_score', 0)
    pick_count = stats.get('pick_count', 0)
    avg_score = stats.get('average_score', 0)

    # 大分数显示
    draw.text((width//2, 320), str(score), font=large_font, fill='#2E7D32', anchor='mm')
    draw.text((width//2, 370), "当前得分", font=small_font, fill='#666666', anchor='mm')

    # 等级 - 去掉emoji
    level_text = get_score_level(score).split()[0]  # 只取文字部分
    draw.text((width//2, 420), level_text, font=normal_font, fill='#FF6F00', anchor='mm')

    # 分隔线
    draw.line([(50, 470), (width-50, 470)], fill='#81C784', width=1)

    # 统计数据
    draw.text((150, 520), f"采摘次数", font=small_font, fill='#666666', anchor='mm')
    draw.text((150, 560), f"{pick_count}", font=normal_font, fill='#1976D2', anchor='mm')

    draw.text((300, 520), f"平均得分", font=small_font, fill='#666666', anchor='mm')
    draw.text((300, 560), f"{avg_score}", font=normal_font, fill='#1976D2', anchor='mm')

    draw.text((450, 520), f"总动作数", font=small_font, fill='#666666', anchor='mm')
    draw.text((450, 560), f"{stats.get('total_actions', 0)}", font=normal_font, fill='#1976D2', anchor='mm')

    # 分隔线
    draw.line([(50, 610), (width-50, 610)], fill='#81C784', width=1)

    # 历史得分
    draw.text((width//2, 650), "最近得分记录", font=small_font, fill='#666666', anchor='mm')
    if scores_history:
        recent = scores_history[-5:]
        history_text = " → ".join([str(s) for s in recent])
        draw.text((width//2, 690), history_text, font=small_font, fill='#333333', anchor='mm')
    else:
        draw.text((width//2, 690), "暂无记录", font=small_font, fill='#999999', anchor='mm')

    # 底部版权
    draw.text((width//2, 760), "© 2026 智茶AI · Tea Picking AI System", font=small_font, fill='#999999', anchor='mm')

    # 保存图片
    filename = f"{user_name}_{mode}_{timestamp}.png"
    filepath = os.path.join(data_dir, filename)
    img.save(filepath, 'PNG')

    # 在页面上显示图片
    st.image(img, caption=f"🎴 {user_name} 的成绩卡", use_container_width=False)
    st.success(f"✅ 成绩卡已保存到: data/{filename}")


if __name__ == "__main__":
    main()

