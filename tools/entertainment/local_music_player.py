# app.py
import streamlit as st
from pathlib import Path

# Config
music_dir = Path.home() / "Music"
supported_exts = [".m4a", ".mp3"]

# Find all audio files recursively
audio_files = sorted([f for ext in supported_exts for f in music_dir.rglob(f"*{ext}")])

# Session state
if "selected_track" not in st.session_state:
    st.session_state.selected_track = None

# Page layout
st.set_page_config(page_title="Spotifly", layout="wide")
st.title("🎧 Spotifly Local")
st.caption(f"Browsing music from `{music_dir}`")

# Sidebar
st.sidebar.header("🔍 Track Browser")
search_query = st.sidebar.text_input("Search tracks or folders")

# Group files by folder (relative to music_dir)
from collections import defaultdict

grouped_tracks = defaultdict(list)
for f in audio_files:
    relative_folder = f.parent.relative_to(music_dir)
    grouped_tracks[relative_folder].append(f)

# Filter groups based on search
filtered_grouped_tracks = {
    folder: [
        track
        for track in tracks
        if search_query.lower() in track.stem.lower()
        or search_query.lower() in str(folder).lower()
    ]
    for folder, tracks in grouped_tracks.items()
}
filtered_grouped_tracks = {k: v for k, v in filtered_grouped_tracks.items() if v}

if not filtered_grouped_tracks:
    st.sidebar.warning("No matching tracks.")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    for folder, tracks in filtered_grouped_tracks.items():
        with st.expander(f"📁 {folder} ({len(tracks)})", expanded=True):
            for i, track_path in enumerate(tracks):
                name = track_path.stem.replace("_", " ").title()
                is_selected = track_path == st.session_state.selected_track
                label = f"✅ {name}" if is_selected else name
                if st.button(label, key=str(track_path)):
                    st.session_state.selected_track = track_path

# Main player section
if st.session_state.selected_track:
    track = st.session_state.selected_track
    title = track.stem.replace("_", " ").title()
    rel_path = track.relative_to(music_dir)
    size_mb = track.stat().st_size / (1024 * 1024)

    st.markdown("---")
    st.subheader(f"🎵 Now Playing: {title}")
    st.caption(f"📁 `{rel_path.parent}` | 💾 {size_mb:.2f} MB")

    with open(track, "rb") as f:
        fmt = "audio/m4a" if track.suffix == ".m4a" else "audio/mp3"
        st.audio(f.read(), format=fmt)
else:
    st.info("Select a track from the sidebar to begin.")
