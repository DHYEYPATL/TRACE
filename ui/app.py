"""
TRACE Operations Console - Streamlit Alternative Frontend.
Provides full dashboarding and control via Streamlit matching Section 4.1 & 10.1.
Communicates directly with FastAPI backend on http://localhost:8000.
"""

import time
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="TRACE | Operations Console",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0c0e14; color: #f8fafc; }
    .kpi-box { background: #181d2a; padding: 12px; border-radius: 8px; border: 1px solid #242b3d; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("TRACE | Trusted Real-time Asset Confidence Engine")
st.caption("L&T Finance × Tech Tatva | Software-only Branch Asset Protection")

# Check backend health
try:
    health_res = requests.get(f"{API_BASE}/api/health", timeout=2.0).json()
    state_res = requests.get(f"{API_BASE}/api/state", timeout=2.0).json()
    backend_online = True
except Exception:
    backend_online = False

if not backend_online:
    st.error("⚠️ TRACE FastAPI Backend is offline. Please launch it with: `python -m trace.main`")
    st.stop()

# 1. Top KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Site Location", state_res.get("site", "demo-branch-01"))
with col2:
    cam_stat = state_res.get("camera_status", "HEALTHY")
    st.metric("Camera Feed", cam_stat, delta="Online" if cam_stat == "HEALTHY" else "-Degraded")
with col3:
    net_stat = state_res.get("network_status", "ONLINE")
    st.metric("Network Sync", net_stat)
with col4:
    fps_val = state_res.get("fps", 5.0)
    st.metric("Edge Processing", f"{fps_val} FPS")

st.markdown("---")

# 2. Main Layout: Video on Left, Assets on Right
v_col, a_col = st.columns([1.6, 1.4])

with v_col:
    st.subheader("📹 Live Camera Feed & Overlays")
    # Stream live MJPEG or image
    st.image(f"{API_BASE}/api/video/feed", caption="Live Video Stream with ROI & Privacy Overlays", use_column_width=True)

with a_col:
    st.subheader("🛡️ Asset Belief Confidence (Doubt Engine)")
    assets = state_res.get("assets", [])
    for a in assets:
        with st.container():
            c_left, c_right = st.columns([3, 1])
            with c_left:
                st.write(f"**{a.get('name', a['id'])}** (`{a['id']}`) - Class: *{a.get('class', 'fixed')}*")
            with c_right:
                state_badge = a.get("state", "SAFE")
                if state_badge == "SAFE":
                    st.success(state_badge)
                elif state_badge in ("MOVED", "MISSING", "SWAPPED"):
                    st.error(state_badge)
                else:
                    st.warning(state_badge)

            belief_val = float(a.get("belief", 1.0))
            st.progress(max(0.0, min(1.0, belief_val)))
            st.caption(f"Confidence: {belief_val:.2f} | Last Verified: {time.strftime('%H:%M:%S', time.localtime(a.get('last_verified_at', time.time())))}")
            st.markdown("<hr style='margin: 4px 0 10px 0;'>", unsafe_allow_html=True)

# 3. Actions & Ledger Verification
st.markdown("---")
b_col1, b_col2, b_col3 = st.columns(3)

with b_col1:
    st.subheader("🛡️ Cryptographic Ledger")
    if st.button("Verify Evidence Ledger"):
        with st.spinner("Checking Ed25519 signatures and SHA-256 hash chains..."):
            v_res = requests.get(f"{API_BASE}/api/ledger/verify").json()
            if v_res.get("ok"):
                st.success(f"✅ {v_res.get('message')}\n\nHead: `{v_res.get('head')[:32]}...`")
            else:
                st.error(f"🚨 TAMPER DETECTED at entry #{v_res.get('bad_idx')}!\n\nReason: {v_res.get('why')}")

with b_col2:
    st.subheader("⚡ Live Demo Fault Injector")
    f_choice = st.selectbox("Select Scenario", [
        ("cover_lens", "Cover Lens (Obstruction)"),
        ("dim_lights", "Dim Lights (Lighting)"),
        ("occlude_asset", "Occlude Safe (Staff In Front)"),
        ("cut_network", "Cut Network (Offline Mode)"),
        ("move_asset", "Move Printer (>15% Drift)"),
        ("remove_asset", "Remove Laptop (Positive Evidence)"),
        ("open_safe_door", "Open Safe Door (State Change)"),
        ("place_unknown_object", "Unknown Object (NEW_OBJECT)"),
        ("swap_asset", "Swap Lookalike (SWAPPED)"),
        ("tamper_ledger", "Tamper Ledger (Simulate Insider Audit Edit)"),
        ("reset", "Reset All Faults"),
    ], format_func=lambda x: x[1])

    if st.button("Trigger Scenario"):
        fault_type = f_choice[0]
        inj_res = requests.post(f"{API_BASE}/api/demo/inject", json={"fault_type": fault_type, "duration_seconds": 6.0}).json()
        st.info(f"Injected: {fault_type}")

with b_col3:
    st.subheader("🔑 Authorize Asset Move")
    auth_asset = st.selectbox("Asset to Authorize", [a["id"] for a in assets])
    auth_mins = st.slider("Authorized Window (Minutes)", 5, 60, 10)
    if st.button("Grant Move Authorization"):
        a_res = requests.post(f"{API_BASE}/api/assets/{auth_asset}/authorize", json={"duration_minutes": auth_mins, "actor": "Streamlit Operator"}).json()
        st.success(a_res.get("message", "Move Authorized."))
