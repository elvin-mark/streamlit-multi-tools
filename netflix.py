from collections import defaultdict
import streamlit as st
from pathlib import Path

# Config
video_dir = Path.home() / "Videos"
supported_exts = [".mp4", ".mkv", ".webm", ".avi", ".mov"]

# Recursively find video files
video_files = sorted([f for ext in supported_exts for f in video_dir.rglob(f"*{ext}")])

# Session state for selected video
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None

# Page layout
st.set_page_config(page_title="NetfliPy", layout="wide")
st.title("🎬 NetfliPy - Your Local Video Player")
st.caption(f"Browsing videos from `{video_dir}`")

# Group videos by folder
grouped_videos = defaultdict(list)
for f in video_files:
    relative_folder = f.parent.relative_to(video_dir)
    grouped_videos[relative_folder].append(f)

# Sidebar search
st.sidebar.header("🔍 Search Videos")
search_query = st.sidebar.text_input("Search videos or folders")

# Filter videos by search
filtered_grouped_videos = {
    folder: [
        video
        for video in videos
        if search_query.lower() in video.stem.lower()
        or search_query.lower() in str(folder).lower()
    ]
    for folder, videos in grouped_videos.items()
}
filtered_grouped_videos = {k: v for k, v in filtered_grouped_videos.items() if v}

if not filtered_grouped_videos:
    st.sidebar.warning("No matching videos.")
    st.stop()

# Sidebar video list
with st.sidebar:
    for folder, videos in filtered_grouped_videos.items():
        with st.expander(f"📁 {folder} ({len(videos)})", expanded=True):
            for video_path in videos:
                video_name = video_path.stem.replace("_", " ").title()
                is_selected = video_path == st.session_state.selected_video
                label = f"✅ {video_name}" if is_selected else video_name
                if st.button(label, key=str(video_path)):
                    st.session_state.selected_video = video_path

# Main video player
if st.session_state.selected_video:
    video = st.session_state.selected_video
    title = video.stem.replace("_", " ").title()
    rel_path = video.relative_to(video_dir)
    size_mb = video.stat().st_size / (1024 * 1024)

    st.markdown("---")
    st.subheader(f"▶️ Now Playing: {title}")
    st.caption(f"📁 `{rel_path.parent}` | 💾 {size_mb:.2f} MB")

    with open(video, "rb") as f:
        video_bytes = f.read()
        st.video(video_bytes)
else:
    st.info("Select a video from the sidebar to begin watching.")
