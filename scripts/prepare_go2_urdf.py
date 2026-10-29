"""Build the workspace Go2 digital-twin URDF from Unitree's official description.

Source: https://github.com/unitreerobotics/unitree_ros  (robots/go2_description)

Changes w.r.t. the original file:
  * ``package://go2_description/dae/*`` mesh URIs -> ``../meshes/*`` (no ROS needed)
  * sensor frames (imu, radar, front_camera) are kept as separate prims when the
    URDF importer merges fixed joints (``dont_collapse="true"``)
  * a horizontal ``nav_lidar`` frame is added on top of the trunk; this is the
    default mount for the simulated planar/2.5D navigation lidar. The original
    ``radar`` frame (Unitree L1 4D lidar, tilted mount) stays available.

Pure python, no Isaac dependency:
    python scripts/prepare_go2_urdf.py
"""

import argparse
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(HERE, "..", "assets", "go2", "urdf")

SENSOR_JOINTS = ("imu_joint", "radar_joint", "front_camera_joint")
NAV_LIDAR_XYZ = "0.18 0 0.095"


def _add_frame(robot: ET.Element, name: str, parent: str, xyz: str, rpy: str = "0 0 0") -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(inertial, "inertia", {k: v for k, v in zip(
        ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"), ("1e-6", "0", "0", "1e-6", "0", "1e-6"))})
    joint = ET.SubElement(robot, "joint", {"name": f"{name}_joint", "type": "fixed", "dont_collapse": "true"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": name})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(URDF_DIR, "go2_unitree_original.urdf"))
    parser.add_argument("--dst", default=os.path.join(URDF_DIR, "go2.urdf"))
    parser.add_argument("--nav-lidar-xyz", default=NAV_LIDAR_XYZ, help="nav_lidar offset from base [m]")
    args = parser.parse_args()

    tree = ET.parse(args.src)
    robot = tree.getroot()
    robot.set("name", "go2")

    for mesh in robot.iter("mesh"):
        fname = mesh.get("filename", "")
        if fname.startswith("package://"):
            mesh.set("filename", "../meshes/" + os.path.basename(fname))

    for joint in robot.iter("joint"):
        if joint.get("name") in SENSOR_JOINTS:
            joint.set("dont_collapse", "true")

    # front_camera is declared without inertial in the original file.
    for link in robot.iter("link"):
        if link.get("name") == "front_camera" and link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": "0.001"})
            ET.SubElement(inertial, "inertia", {"ixx": "1e-6", "ixy": "0", "ixz": "0",
                                                "iyy": "1e-6", "iyz": "0", "izz": "1e-6"})

    if not any(l.get("name") == "nav_lidar" for l in robot.iter("link")):
        _add_frame(robot, "nav_lidar", "base", args.nav_lidar_xyz)

    ET.indent(tree, space="  ")
    tree.write(args.dst, encoding="utf-8", xml_declaration=True)
    print(f"[prepare_go2_urdf] wrote {os.path.abspath(args.dst)}")


if __name__ == "__main__":
    main()
