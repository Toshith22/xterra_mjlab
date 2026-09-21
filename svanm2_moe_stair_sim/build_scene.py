"""Procedural scene generator for SvanM2 on multi-stair terrains in MuJoCo.

Generates 6 distinct 12-step staircases:
- 10 cm (0.10 m)
- 12 cm (0.12 m)
- 14 cm (0.14 m)
- 15 cm (0.15 m)
- 16 cm (0.16 m)
- 20 cm (0.20 m)

Each staircase has:
- A flat approach runway (length 4.0 m)
- Exactly 12 steps (tread 0.30 m, width 2.0 m) with alternating light/dark bands
- A wide top turnaround platform (length 3.0 m, width 2.4 m) flush with the 12th step
- Visual marker pillars and color-coded theme
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE.parent / "xterra_mjlab").exists():
    REPO_ROOT = HERE.parent
elif (HERE.parent / "m2_public_training").exists():
    REPO_ROOT = HERE.parent / "m2_public_training"
else:
    REPO_ROOT = HERE.parent

SVAN_XML = REPO_ROOT / "xterra_mjlab/assets/svanm2/xml/svanm2_mjlab.xml"
MESH_DIR = REPO_ROOT / "xterra_mjlab/assets/svanm2/meshes"

# 6 stair heights as requested by user
TRACK_CONFIGS = [
    {"index": 1, "height": 0.10, "label": "10cm", "color_main": (0.15, 0.55, 0.80, 1.0), "color_alt": (0.25, 0.65, 0.90, 1.0)},
    {"index": 2, "height": 0.12, "label": "12cm", "color_main": (0.10, 0.65, 0.60, 1.0), "color_alt": (0.20, 0.75, 0.70, 1.0)},
    {"index": 3, "height": 0.14, "label": "14cm", "color_main": (0.20, 0.70, 0.30, 1.0), "color_alt": (0.30, 0.80, 0.40, 1.0)},
    {"index": 4, "height": 0.15, "label": "15cm", "color_main": (0.90, 0.55, 0.10, 1.0), "color_alt": (0.95, 0.65, 0.20, 1.0)},
    {"index": 5, "height": 0.16, "label": "16cm", "color_main": (0.85, 0.30, 0.20, 1.0), "color_alt": (0.95, 0.40, 0.30, 1.0)},
    {"index": 6, "height": 0.20, "label": "20cm", "color_main": (0.65, 0.20, 0.75, 1.0), "color_alt": (0.75, 0.30, 0.85, 1.0)},
]

NUM_STEPS = 12
TREAD_DEPTH = 0.30  # 30 cm step tread
STAIR_WIDTH = 2.00  # 2.0 m wide stairs
PLATFORM_LEN = 3.00 # 3.0 m top platform length
PLATFORM_WID = 2.40 # 2.4 m top platform width
APPROACH_LEN = 4.00 # 4.0 m approach runway
TRACK_SPACING = 3.80 # 3.8 m between track centerlines

JOINTS = [f"{leg}_{stage}_joint" for leg in ("FL", "FR", "RL", "RR") for stage in ("hip", "thigh", "calf")]
STAND_Q = np.array([0., 0.5806, -1.1716, 0., 0.5806, -1.1716, 0., 0.7167, -1.1342, 0., 0.7167, -1.1342], dtype=np.float32)

# Simulation joint limits: allow natural forward leg extension (calf up to +0.50 rad)
SIM_JOINT_LIMITS = np.array([
    [-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, 0.50],  # FL
    [-1.0472, 1.0472], [-1.5708, 3.4907], [-2.7227, 0.50],  # FR
    [-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, 0.50],  # RL
    [-1.0472, 1.0472], [-0.5236, 4.5379], [-2.7227, 0.50]   # RR
])

def get_track_y(track_idx_0based: int) -> float:
    """Centerline Y coordinate of a track (0 to 5)."""
    return (track_idx_0based - 2.5) * TRACK_SPACING

def get_spawn_pos(track_idx_1based: int, on_top: bool = False) -> tuple[float, float, float]:
    """Get nominal spawn position (X, Y, Z) for a given track."""
    idx0 = track_idx_1based - 1
    y = get_track_y(idx0)
    cfg = TRACK_CONFIGS[idx0]
    if on_top:
        # Middle of top platform, facing backwards (-X)
        x = NUM_STEPS * TREAD_DEPTH + PLATFORM_LEN * 0.5
        z = NUM_STEPS * cfg["height"] + 0.365
        return (x, y, z)
    else:
        # On approach pad facing forwards (+X), standing firmly on floor (Z=0)
        x = -1.50
        z = 0.365
        return (x, y, z)

def build_scene_xml(output_path: Path = HERE / "scene.xml") -> Path:
    """Build and save the complete MuJoCo scene XML."""
    assert SVAN_XML.exists(), f"Source XML missing: {SVAN_XML}"
    raw_xml = SVAN_XML.read_text()
    clean_xml = re.sub(r"<!--.*?-->", "", raw_xml, flags=re.S)
    root = ET.fromstring(clean_xml)

    # Compiler settings
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", os.path.relpath(MESH_DIR, output_path.parent))
    compiler.set("angle", "radian")

    # Options matching 200 Hz physics
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.attrib.update({
        "timestep": "0.005",
        "integrator": "implicitfast",
        "cone": "elliptic",
        "impratio": "100",
        "iterations": "20",
        "ls_iterations": "20",
    })

    # Assets & textures
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    
    # Textures for ground and sky
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.35 0.55 0.75", rgb2="0.1 0.15 0.2", width="512", height="512")
    ET.SubElement(asset, "texture", type="2d", name="grid_plane", builtin="checker", mark="edge",
                  rgb1="0.22 0.25 0.28", rgb2="0.16 0.18 0.20", markrgb="0.45 0.50 0.55", width="300", height="300")
    ET.SubElement(asset, "material", name="ground_mat", texture="grid_plane", texuniform="true", texrepeat="40 40", reflectance="0.15")
    
    ET.SubElement(asset, "texture", type="2d", name="runway_tex", builtin="flat", rgb1="0.30 0.32 0.35", rgb2="0.30 0.32 0.35", width="64", height="64")
    ET.SubElement(asset, "material", name="runway_mat", texture="runway_tex", reflectance="0.05")

    # Clear old actuators and add torque actuators for SvanM2
    actuators = root.find("actuator")
    if actuators is not None:
        actuators.clear()
    else:
        actuators = ET.SubElement(root, "actuator")

    for jname in JOINTS:
        act_name = jname.removesuffix("_joint")
        limit = "24" if "calf" in jname else "12"
        ET.SubElement(actuators, "motor", name=act_name, joint=jname, ctrllimited="true", ctrlrange=f"-{limit} {limit}", gear="1")

    # Foot geom tuning: firm contact, realistic friction, symmetric contact
    for geom in root.findall(".//geom"):
        gname = geom.get("name", "")
        if gname in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            geom.attrib.update({
                "condim": "3",
                "priority": "1",
                "friction": "0.85 0.01 0.001",
                "contype": "1",
                "conaffinity": "1",
                "solimp": "0.9 0.95 0.001",
                "solref": "0.002 1",
            })

    # Set joint limits on robot
    for j_elem in root.findall(".//joint"):
        jname = j_elem.get("name", "")
        if jname in JOINTS:
            idx = JOINTS.index(jname)
            lo, hi = SIM_JOINT_LIMITS[idx]
            j_elem.set("limited", "true")
            j_elem.set("range", f"{lo:.5f} {hi:.5f}")

    # Build worldbody with stairs
    wb = root.find("worldbody")
    assert wb is not None

    # Lighting
    ET.SubElement(wb, "light", name="sun1", pos="5 0 6", dir="-0.5 0 -1", diffuse="0.75 0.75 0.75", ambient="0.28 0.28 0.28", specular="0.1 0.1 0.1", directional="true")
    ET.SubElement(wb, "light", name="sun2", pos="-5 0 6", dir="0.5 0 -1", diffuse="0.45 0.45 0.45", specular="0 0 0", directional="true")

    # Infinite ground plane
    ET.SubElement(wb, "geom", name="floor", type="plane", size="80 80 0.1", pos="0 0 0", material="ground_mat", friction="0.8 0.005 0.0001", condim="3")

    # Generate each stair track
    for t_idx, cfg in enumerate(TRACK_CONFIGS):
        track_y = get_track_y(t_idx)
        step_h = cfg["height"]
        t_label = cfg["label"]
        col_main = " ".join(f"{c:.3f}" for c in cfg["color_main"])
        col_alt = " ".join(f"{c:.3f}" for c in cfg["color_alt"])
        col_dark = " ".join(f"{c*0.6:.3f}" for c in cfg["color_main"][:3]) + " 1.0"
        
        # Track Body container
        t_body = ET.SubElement(wb, "body", name=f"track_{t_idx+1}_{t_label}", pos=f"0 {track_y:.3f} 0")

        # 1. Approach runway (thin visual marker flush with floor, zero-collision so it does not conflict with floor plane)
        runway_x = -APPROACH_LEN * 0.5
        ET.SubElement(t_body, "geom", name=f"runway_{t_idx+1}", type="box",
                      size=f"{APPROACH_LEN*0.5:.3f} {PLATFORM_WID*0.5:.3f} 0.001",
                      pos=f"{runway_x:.3f} 0 0.001",
                      material="runway_mat", contype="0", conaffinity="0")

        # Visual entrance marker pillars (left and right)
        ET.SubElement(t_body, "geom", name=f"marker_L_{t_idx+1}", type="cylinder",
                      size="0.10 0.40", pos=f"-2.5 -{PLATFORM_WID*0.55:.3f} 0.40",
                      rgba=col_main, contype="0", conaffinity="0")
        ET.SubElement(t_body, "geom", name=f"marker_R_{t_idx+1}", type="cylinder",
                      size="0.10 0.40", pos=f"-2.5 {PLATFORM_WID*0.55:.3f} 0.40",
                      rgba=col_main, contype="0", conaffinity="0")
        # Entrance crossbar / banner geometry indicating height
        ET.SubElement(t_body, "geom", name=f"banner_{t_idx+1}", type="box",
                      size=f"0.08 {PLATFORM_WID*0.55:.3f} 0.12", pos=f"-2.5 0 0.85",
                      rgba=col_main, contype="0", conaffinity="0")

        # 2. 12 steps
        for s in range(NUM_STEPS):
            step_num = s + 1
            cur_h = step_num * step_h
            step_x_center = (s + 0.5) * TREAD_DEPTH
            step_half_x = TREAD_DEPTH * 0.5
            step_half_y = STAIR_WIDTH * 0.5
            step_half_z = cur_h * 0.5
            
            # Alternating step colors for visual depth perception
            rgba = col_alt if (s % 2 == 0) else col_main
            
            ET.SubElement(t_body, "geom", name=f"step_{t_idx+1}_{step_num}", type="box",
                          size=f"{step_half_x:.4f} {step_half_y:.4f} {step_half_z:.4f}",
                          pos=f"{step_x_center:.4f} 0 {step_half_z:.4f}",
                          rgba=rgba, friction="0.90 0.005 0.0001", condim="3")

        # 3. Top turnaround platform
        stair_run_total = NUM_STEPS * TREAD_DEPTH
        plat_top_z = NUM_STEPS * step_h
        plat_x_center = stair_run_total + PLATFORM_LEN * 0.5
        plat_half_z = plat_top_z * 0.5

        ET.SubElement(t_body, "geom", name=f"platform_{t_idx+1}", type="box",
                      size=f"{PLATFORM_LEN*0.5:.3f} {PLATFORM_WID*0.5:.3f} {plat_half_z:.4f}",
                      pos=f"{plat_x_center:.4f} 0 {plat_half_z:.4f}",
                      rgba=col_dark, friction="0.90 0.005 0.0001", condim="3")

        # Safety side guide curbs on platform (to prevent falling off edges while turning)
        curb_h = 0.08
        # Left curb on platform
        ET.SubElement(t_body, "geom", name=f"curb_L_{t_idx+1}", type="box",
                      size=f"{PLATFORM_LEN*0.5:.3f} 0.04 {curb_h*0.5:.3f}",
                      pos=f"{plat_x_center:.4f} -{PLATFORM_WID*0.5+0.04:.3f} {plat_top_z + curb_h*0.5:.3f}",
                      rgba=col_main, friction="0.4 0.001 0.0001", condim="3")
        # Right curb on platform
        ET.SubElement(t_body, "geom", name=f"curb_R_{t_idx+1}", type="box",
                      size=f"{PLATFORM_LEN*0.5:.3f} 0.04 {curb_h*0.5:.3f}",
                      pos=f"{plat_x_center:.4f} {PLATFORM_WID*0.5+0.04:.3f} {plat_top_z + curb_h*0.5:.3f}",
                      rgba=col_main, friction="0.4 0.001 0.0001", condim="3")
        # Back wall on platform
        ET.SubElement(t_body, "geom", name=f"curb_back_{t_idx+1}", type="box",
                      size=f"0.06 {PLATFORM_WID*0.5+0.08:.3f} {curb_h*0.5:.3f}",
                      pos=f"{stair_run_total + PLATFORM_LEN + 0.06:.3f} 0 {plat_top_z + curb_h*0.5:.3f}",
                      rgba=col_main, friction="0.4 0.001 0.0001", condim="3")

    # Initial robot spawn at Track 1 (10cm) approach pad
    init_x, init_y, init_z = get_spawn_pos(1, on_top=False)
    base = wb.find(".//body[@name='base']")
    if base is not None:
        base.set("pos", f"{init_x:.3f} {init_y:.3f} {init_z:.3f}")
        # Ensure freejoint exists
        fj = base.find("freejoint")
        if fj is None and not any(j.get("type") == "free" for j in base.findall("joint")):
            base.insert(0, ET.Element("freejoint", name="floating_base"))

    # Initial standing pose keyframe so robot is upright immediately upon model load
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    else:
        keyframe.clear()
    stand_qpos_str = f"{init_x:.3f} {init_y:.3f} {init_z:.3f} 1 0 0 0 " + " ".join(f"{val:.5f}" for val in STAND_Q)
    ET.SubElement(keyframe, "key", name="home", qpos=stand_qpos_str)

    # Write formatted XML
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    output_path.write_text(xml_str)
    print(f"Successfully generated {output_path} with {len(TRACK_CONFIGS)} stair tracks ({NUM_STEPS} steps each).")
    return output_path

if __name__ == "__main__":
    build_scene_xml()
