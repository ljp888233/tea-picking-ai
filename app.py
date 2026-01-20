"""
AI采茶动作捕捉系统 V2.0 - WebRTC云端完整版
支持四种模式：体验/效率/质控/教学
包含：动作捕捉、实时反馈、成绩导出
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

# WebRTC 配置 - 使用多个 STUN/TURN 服务器提高连接成功率
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]}
)

# 页面配置
st.set_page_config(page_title="智茶 AI", page_icon="🍵", layout="wide", initial_sidebar_state="expanded")

# CSS样式
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #E8F5E9 100%); }
    .main-title { text-align: center; font-size: 2.5rem; font-weight: 700; background: linear-gradient(120deg, #2E7D32 0%, #00695C 50%, #1B5E20 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .sub-title { text-align: center; color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .score-display { font-size: 4rem; font-weight: 800; text-align: center; text-shadow: 0 0 20px currentColor; }
    .big-number { font-size: 3rem; font-weight: 700; text-align: center; color: #1976D2; }
    .mode-title { font-size: 1.1rem; color: #37474F; padding: 0.8rem; border-radius: 10px; background: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 100%); border-left: 4px solid #00897B; margin-bottom: 1rem; }
    .achievement-badge { display: inline-block; padding: 0.4rem 1rem; margin: 0.3rem; border-radius: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 0.85rem; }
    .feedback-item { padding: 0.8rem; margin: 0.4rem 0; border-radius: 10px; background: #FAFAFA; border-left: 4px solid #4CAF50; }
    .warning-box { padding: 1rem; border-radius: 12px; background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); border-left: 4px solid #FF9800; }
    .success-box { padding: 1rem; border-radius: 12px; background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); border-left: 4px solid #4CAF50; }
    .teaching-step { padding: 1rem; margin: 0.5rem 0; border-radius: 12px; background: linear-gradient(145deg, #E3F2FD 0%, #BBDEFB 100%); }
    .tech-card { background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid rgba(46,125,50,0.1); margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


class VideoProcessor:
    """视频处理器 - 处理每一帧并进行动作分析"""
    lock = threading.Lock()
    score = 0
    feedback = []
    stats = {'pick_count': 0, 'current_score': 0, 'average_score': 0, 'total_actions': 0}
    start_time = time.time()
    scores_history = []

    def __init__(self):
        self.pose_detector = PoseDetector()
        self.hand_detector = HandDetector()
        self.analyzer = TeaPickingAnalyzer()
        self.show_pose = True
        self.show_hands = True
        self.show_fps = True
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

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
            with VideoProcessor.lock:
                VideoProcessor.score = result['score']
                VideoProcessor.feedback = result['feedback']
                VideoProcessor.stats = self.analyzer.get_statistics()
                if result['score'] > 0 and (len(VideoProcessor.scores_history) == 0 or VideoProcessor.scores_history[-1] != result['score']):
                    VideoProcessor.scores_history.append(result['score'])
                    if len(VideoProcessor.scores_history) > 100:
                        VideoProcessor.scores_history = VideoProcessor.scores_history[-100:]

        # FPS计算
        self.frame_count += 1
        if self.frame_count >= 10:
            self.fps = self.frame_count / (time.time() - self.fps_time)
            self.fps_time = time.time()
            self.frame_count = 0

        # 在画面上显示信息
        if self.show_fps:
            cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        score_color = get_score_color(VideoProcessor.score)
        cv2.putText(img, f"Score: {VideoProcessor.score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)

        # 显示等级
        level = get_score_level(VideoProcessor.score)
        cv2.putText(img, level, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def rgb_to_hex(bgr):
    """BGR颜色转十六进制"""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def reset_stats():
    """重置统计数据"""
    with VideoProcessor.lock:
        VideoProcessor.score = 0
        VideoProcessor.feedback = []
        VideoProcessor.stats = {'pick_count': 0, 'current_score': 0, 'average_score': 0, 'total_actions': 0}
        VideoProcessor.start_time = time.time()
        VideoProcessor.scores_history = []


def export_score_card(user_name, mode):
    """导出成绩卡片"""
    with VideoProcessor.lock:
        stats = VideoProcessor.stats.copy()
        scores_history = VideoProcessor.scores_history.copy()
        score = VideoProcessor.score

    if not user_name:
        st.warning("⚠️ 请先在侧边栏输入您的姓名！")
        return

    # 创建data文件夹
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    width, height = 600, 800
    img = Image.new('RGB', (width, height), '#E8F5E9')
    draw = ImageDraw.Draw(img)

    # 渐变背景
    for y in range(height):
        r = int(232 - (y / height) * 30)
        g = int(245 - (y / height) * 20)
        b = int(233 - (y / height) * 30)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 边框
    draw.rectangle([20, 20, width-20, height-20], outline='#2E7D32', width=3)

    # 字体
    try:
        title_font = ImageFont.truetype("msyh.ttc", 36)
        large_font = ImageFont.truetype("msyh.ttc", 48)
        normal_font = ImageFont.truetype("msyh.ttc", 24)
        small_font = ImageFont.truetype("msyh.ttc", 18)
    except:
        title_font = large_font = normal_font = small_font = ImageFont.load_default()

    # 标题
    draw.text((width//2, 60), "智茶AI", font=title_font, fill='#1B5E20', anchor='mm')
    draw.text((width//2, 100), "- 采茶成绩卡 -", font=normal_font, fill='#2E7D32', anchor='mm')
    draw.line([(50, 140), (width-50, 140)], fill='#81C784', width=2)

    # 用户信息
    draw.text((width//2, 180), f"使用者: {user_name}", font=normal_font, fill='#333333', anchor='mm')
    draw.text((width//2, 220), datetime.now().strftime("%Y年%m月%d日 %H:%M"), font=small_font, fill='#666666', anchor='mm')

    # 分数
    draw.text((width//2, 320), str(score), font=large_font, fill='#2E7D32', anchor='mm')
    draw.text((width//2, 370), "当前得分", font=small_font, fill='#666666', anchor='mm')
    level_text = get_score_level(score).split()[0]
    draw.text((width//2, 420), level_text, font=normal_font, fill='#FF6F00', anchor='mm')

    draw.line([(50, 470), (width-50, 470)], fill='#81C784', width=1)

    # 统计
    draw.text((150, 520), "采摘次数", font=small_font, fill='#666666', anchor='mm')
    draw.text((150, 560), str(stats.get('pick_count', 0)), font=normal_font, fill='#1976D2', anchor='mm')
    draw.text((300, 520), "平均得分", font=small_font, fill='#666666', anchor='mm')
    draw.text((300, 560), str(stats.get('average_score', 0)), font=normal_font, fill='#1976D2', anchor='mm')
    draw.text((450, 520), "总动作数", font=small_font, fill='#666666', anchor='mm')
    draw.text((450, 560), str(stats.get('total_actions', 0)), font=normal_font, fill='#1976D2', anchor='mm')

    draw.line([(50, 610), (width-50, 610)], fill='#81C784', width=1)

    # 历史
    draw.text((width//2, 650), "最近得分记录", font=small_font, fill='#666666', anchor='mm')
    if scores_history:
        recent = scores_history[-5:]
        draw.text((width//2, 690), " → ".join([str(s) for s in recent]), font=small_font, fill='#333333', anchor='mm')
    else:
        draw.text((width//2, 690), "暂无记录", font=small_font, fill='#999999', anchor='mm')

    draw.text((width//2, 760), "© 2026 智茶AI", font=small_font, fill='#999999', anchor='mm')

    # 保存
    filename = f"{user_name}_{mode}_{timestamp}.png"
    filepath = os.path.join(data_dir, filename)
    img.save(filepath, 'PNG')
    st.image(img, caption=f"🎴 {user_name} 的成绩卡", use_container_width=False)
    st.success(f"✅ 成绩卡已保存到: data/{filename}")



def main():
    st.markdown('<h1 class="main-title">🍵 智茶 AI · 采茶动作捕捉系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">🌿 传承千年茶艺，智能科技赋能 | AI-Powered Tea Picking</p>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.markdown('<div style="text-align:center;"><span style="font-size:3rem;">🍵</span><h2 style="color:#2E7D32;">智茶 AI</h2></div>', unsafe_allow_html=True)
        st.divider()

        st.subheader("👤 使用者信息")
        user_name = st.text_input("姓名", placeholder="请输入您的姓名", key="user_name")
        if not user_name:
            st.caption("⚠️ 请输入姓名以便导出数据")

        st.divider()
        st.subheader("🎯 模式选择")
        mode = st.selectbox("选择模式", ["🎮 体验模式", "📊 效率模式", "✅ 质控模式", "📚 教学模式"], label_visibility="collapsed")

        st.divider()
        st.subheader("👁️ 显示选项")
        show_pose = st.checkbox("显示身体骨骼", value=True)
        show_hands = st.checkbox("显示手部骨骼", value=True)
        show_fps = st.checkbox("显示帧率", value=True)

        st.divider()
        if st.button("🔄 重置统计", use_container_width=True):
            reset_stats()
            st.success("✅ 统计已重置！")

        st.markdown('<p style="text-align:center;color:#999;font-size:0.8rem;">Version 2.0 WebRTC<br>© 2026 智茶AI</p>', unsafe_allow_html=True)

    # 根据模式渲染
    if mode == "🎮 体验模式":
        render_experience_mode(user_name, show_pose, show_hands, show_fps)
    elif mode == "📊 效率模式":
        render_efficiency_mode(user_name, show_pose, show_hands, show_fps)
    elif mode == "✅ 质控模式":
        render_quality_mode(user_name, show_pose, show_hands, show_fps)
    elif mode == "📚 教学模式":
        render_teaching_mode(user_name, show_pose, show_hands, show_fps)



def render_experience_mode(user_name, show_pose, show_hands, show_fps):
    """🎮 体验模式"""
    st.markdown('<p class="mode-title">🎮 体验模式 - 趣味互动，挑战采茶大师！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时画面")
        st.info("👆 点击 START 开启摄像头，首次使用请允许浏览器访问摄像头权限")

        webrtc_streamer(
            key="experience",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.button("🎴 生成成绩卡", use_container_width=True, key="exp_export"):
            export_score_card(user_name, "experience")

    with col2:
        st.subheader("🏆 实时成绩")
        with VideoProcessor.lock:
            score = VideoProcessor.score
            stats = VideoProcessor.stats.copy()

        score_color = rgb_to_hex(get_score_color(score))
        st.markdown(f'<p class="score-display" style="color:{score_color}">{score}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;font-size:1.5rem;">{get_score_level(score)}</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("🎖️ 成就徽章")
        achievements = []
        if stats.get('pick_count', 0) >= 1: achievements.append("🌱 初次采摘")
        if stats.get('pick_count', 0) >= 10: achievements.append("🍃 采茶新秀")
        if stats.get('pick_count', 0) >= 50: achievements.append("🌿 采茶达人")
        if stats.get('average_score', 0) >= 80: achievements.append("⭐ 高分选手")

        if achievements:
            st.markdown("".join([f'<span class="achievement-badge">{a}</span>' for a in achievements]), unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#999;">继续努力解锁成就！</span>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 统计数据")
        st.markdown(f"""
        - 🍃 采摘次数: **{stats.get('pick_count', 0)}**
        - 📊 当前评分: **{stats.get('current_score', 0)}**
        - 📈 平均评分: **{stats.get('average_score', 0)}**
        """)

        st.divider()
        st.subheader("💡 实时反馈")
        with VideoProcessor.lock:
            feedback = VideoProcessor.feedback.copy()
        if feedback:
            for fb in feedback:
                st.markdown(f'<div class="feedback-item">{fb}</div>', unsafe_allow_html=True)
        else:
            st.info("等待检测手部动作...")



def render_efficiency_mode(user_name, show_pose, show_hands, show_fps):
    """📊 效率模式"""
    st.markdown('<p class="mode-title">📊 效率模式 - 统计采摘效率，提升工作表现！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 实时监控")
        st.info("👆 点击 START 开启摄像头")

        webrtc_streamer(
            key="efficiency",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.button("🎴 生成成绩卡", use_container_width=True, key="eff_export"):
            export_score_card(user_name, "efficiency")

    with col2:
        st.subheader("⏱️ 效率数据")

        with VideoProcessor.lock:
            stats = VideoProcessor.stats.copy()
            elapsed = time.time() - VideoProcessor.start_time

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**采摘次数**")
            st.markdown(f'<p class="big-number">{stats.get("pick_count", 0)}</p>', unsafe_allow_html=True)
        with col_b:
            st.markdown("**每分钟速度**")
            speed = stats.get('pick_count', 0) / (elapsed / 60) if elapsed > 0 else 0
            st.markdown(f'<p class="big-number">{speed:.1f}</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📈 效率趋势")
        st.progress(min(stats.get('pick_count', 0) / 100, 1.0), text=f"目标: 100次")

        st.divider()
        st.subheader("📋 详细统计")
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        st.markdown(f"""
        - ⏱️ 已用时间: **{minutes}分{seconds}秒**
        - 🎯 采摘次数: **{stats.get('pick_count', 0)}**
        - 📈 平均速度: **{speed:.1f}次/分钟**
        - 💯 平均质量: **{stats.get('average_score', 0)}分**
        """)

        st.divider()
        st.subheader("💡 实时反馈")
        with VideoProcessor.lock:
            feedback = VideoProcessor.feedback.copy()
        if feedback:
            for fb in feedback:
                st.markdown(f'<div class="feedback-item">{fb}</div>', unsafe_allow_html=True)
        else:
            st.info("等待检测手部动作...")



def render_quality_mode(user_name, show_pose, show_hands, show_fps):
    """✅ 质控模式"""
    st.markdown('<p class="mode-title">✅ 质控模式 - 规范动作，保证茶叶品质！</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 动作监控")
        st.info("👆 点击 START 开启摄像头")

        webrtc_streamer(
            key="quality",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.button("🎴 生成成绩卡", use_container_width=True, key="qc_export"):
            export_score_card(user_name, "quality")

    with col2:
        st.subheader("📋 质量评估")
        with VideoProcessor.lock:
            score = VideoProcessor.score
            stats = VideoProcessor.stats.copy()
            feedback = VideoProcessor.feedback.copy()

        quality_level = "优秀 ✅" if score >= 80 else "良好 👍" if score >= 60 else "需改进 ⚠️"
        quality_color = "#4caf50" if score >= 80 else "#ff9800" if score >= 60 else "#f44336"
        st.markdown(f'<p style="font-size:2rem;text-align:center;color:{quality_color}">{quality_level}</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("⚠️ 实时提醒")
        warnings = [fb for fb in feedback if '✗' in fb or '△' in fb]
        if warnings:
            st.markdown('<div class="warning-box">' + '<br>'.join(warnings) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ 动作规范，继续保持！</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("✅ 规范检查项")
        st.markdown(f"""
        - {'✅' if score >= 70 else '❌'} 捏取姿势规范
        - {'✅' if score >= 60 else '❌'} 手指姿态自然
        - {'✅' if score >= 50 else '❌'} 动作稳定流畅
        """)

        st.divider()
        st.subheader("📊 质量统计")
        good_rate = (stats.get('average_score', 0) / 100) * 100
        st.markdown(f"""
        - 📊 合格率: **{good_rate:.1f}%**
        - 🔢 检测次数: **{stats.get('total_actions', 0)}**
        - 📈 平均得分: **{stats.get('average_score', 0)}**
        """)



def render_teaching_mode(user_name, show_pose, show_hands, show_fps):
    """📚 教学模式"""
    st.markdown('<p class="mode-title">📚 教学模式 - 学习标准采茶技艺！</p>', unsafe_allow_html=True)

    # 教学步骤
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
        st.info("👆 点击 START 开启摄像头")

        webrtc_streamer(
            key="teaching",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if st.button("🎴 生成成绩卡", use_container_width=True, key="teach_export"):
            export_score_card(user_name, "teaching")

    with col2:
        st.subheader("📝 动作评价")
        with VideoProcessor.lock:
            score = VideoProcessor.score
            stats = VideoProcessor.stats.copy()
            feedback = VideoProcessor.feedback.copy()

        score_color = rgb_to_hex(get_score_color(score))
        grade = "优秀" if score >= 80 else "良好" if score >= 60 else "继续练习"
        st.markdown(f'<p style="font-size:2.5rem;text-align:center;color:{score_color}">{score}分 - {grade}</p>', unsafe_allow_html=True)

        st.divider()
        st.subheader("💡 改进建议")
        if feedback:
            for fb in feedback:
                st.markdown(f'<div class="feedback-item">{fb}</div>', unsafe_allow_html=True)
        else:
            st.info("等待检测手部动作...")

        st.divider()
        st.subheader("📈 学习进度")
        progress_pct = min(stats.get('average_score', 0) / 100, 1.0)
        st.progress(progress_pct, text=f"掌握程度: {int(progress_pct*100)}%")


if __name__ == "__main__":
    main()
