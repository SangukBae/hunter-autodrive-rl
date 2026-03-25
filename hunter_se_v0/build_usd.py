# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hunter SE V0 USD 생성 스크립트.

기본 도형(Box/Cylinder)만 사용하여 Hunter SE v0 로봇의 USD articulation 파일을
pxr Python API로 생성합니다.

생성 파일: hunter_se_v0.usda (이 스크립트와 같은 디렉토리)

단독 실행:
    /workspace/isaaclab/isaaclab.sh -p hunter_se_v0/build_usd.py

물리 파라미터 출처:
    hunter_se URDF / Physics.usda   — 관절 위치·방향, 질량 분배

시각 형상 출처:
    /robot_isaac/ugv_gazebo_sim/hunter_se/hunter_se_description/meshes/*.STL
    → 별도 hunter_se_v0_meshes.usdc (binary) 로 빌드 후 참조

계층구조 설계 (플랫 방식):
    PhysX는 RigidBody 아래에 또 다른 RigidBody가 중첩될 경우
    !resetXformStack! 을 요구합니다. 이를 피하기 위해 모든 링크를
    루트 Xform(HunterSEV0)의 직접 자식으로 배치합니다 (플랫 계층).

        /HunterSEV0                  ← Xform + ArticulationRootAPI
          /base_link                 ← 차체
          /fr_steer_left_link        ← 전륜 좌 너클
          /fr_left_link              ← 전륜 좌 바퀴
          /fr_steer_right_link       ← 전륜 우 너클
          /fr_right_link             ← 전륜 우 바퀴
          /re_left_link              ← 후륜 좌 바퀴
          /re_right_link             ← 후륜 우 바퀴
          /Physics                   ← 조인트 Scope

    조인트는 body0/body1 레퍼런스로 트리를 정의합니다 (USD 계층과 무관).

관절 축 설계:
    - 조향 관절: axis=Z  (수직축 회전)
    - 바퀴 관절: axis=Y  (횡방향 회전)
      양수 속도 목표 → 전진 방향 (+X), ackermann.py와 동일 부호 규약
"""

import os
import struct

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

try:
    from pxr import PhysxSchema
    _PHYSX = True
except ImportError:
    _PHYSX = False

try:
    import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
    _RTX_LIDAR = True
except ImportError:
    IsaacSensorSchema = None
    _RTX_LIDAR = False

# ── 기구 상수 (hunter_se URDF Physics.usda 정확한 값) ─────────────────────────
FRONT_AX_X   =  0.34058   # 전륜 관절 X [m]
FRONT_AX_Y   =  0.24619   # 전륜 조향 피벗 ±Y [m]
FRONT_AX_Z   = -0.1535    # 전륜 관절 Z [m]
REAR_AX_X    = -0.2078    # 후륜 관절 X [m]
REAR_AX_Y    =  0.252     # 후륜 허브 ±Y [m]
REAR_AX_Z    = -0.158     # 후륜 관절 Z [m]

WHEEL_RADIUS =  0.1375    # 바퀴 반지름 [m]  (매뉴얼 직경 0.275 m / 2)
WHEEL_WIDTH  =  0.080     # 바퀴 폭 [m]
MAX_STEER_DEG = 22.0      # 최대 조향각 [deg]

LIDAR_Z       =  0.50      # base_link 기준 LiDAR 높이 [m]
LIDAR_RADIUS  =  0.0425    # Ouster OS1-32 근사 반지름 [m]
LIDAR_HEIGHT  =  0.073     # Ouster OS1-32 근사 높이 [m]

# 차체 외형 (매뉴얼 기준)
CHASSIS_L    =  0.817
CHASSIS_W    =  0.640    # 매뉴얼 전체 폭 0.640~0.644 m 하한 적용
CHASSIS_H    =  0.120    # 차체 박스 높이만 반영 (전체 외형 0.304~0.310 m 아님)

# 질량 (공차중량 42 kg 기준 역산)
# 너클 2 × 3.149 + 바퀴 4 × 3.149 = 18.894 kg
# CHASSIS_MASS = 42.0 − 18.894 = 23.106 kg
CHASSIS_MASS = 23.106
KNUCKLE_MASS =  3.149
WHEEL_MASS   =  3.149

# STL 메시 경로 (Gazebo 패키지 기준)
_MESH_DIR = "/robot_isaac/ugv_gazebo_sim/hunter_se/hunter_se_description/meshes"

# 시각화 색상 (Hunter SE 도장색 근사)
_COLOR_BODY  = Gf.Vec3f(0.18, 0.22, 0.14)   # 올리브 그린 (차체)
_COLOR_WHEEL = Gf.Vec3f(0.10, 0.10, 0.10)   # 다크 그레이 (바퀴/너클)

# 바퀴 링크의 URDF joint rpy → USD 시각 메시 보정 쿼터니언 (w, x, y, z)
# URDF 조인트가 Rx(±90°)로 바퀴를 회전시키므로, 시각 메시에 동일 회전 적용
_Q_RX_POS = Gf.Quatf(0.7071068, 0.7071068, 0.0, 0.0)   # Rx(+90°)
_Q_RX_NEG = Gf.Quatf(0.7071068, -0.7071068, 0.0, 0.0)  # Rx(-90°)


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _identity():
    return Gf.Quatf(1.0, 0.0, 0.0, 0.0)


def _apply_rigid_body(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.GetMassAttr().Set(float(mass))
    if _PHYSX:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)


def _apply_collision(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    if _PHYSX:
        PhysxSchema.PhysxCollisionAPI.Apply(prim)


def _make_link(stage, path, mass, world_pos):
    """플랫 계층용 Rigid body Xform 링크 생성.

    world_pos: articulation root 좌표계 기준 초기 위치 [m]
    모든 링크는 루트의 직접 자식이므로 resetXformStack 불필요.
    """
    xform = UsdGeom.Xform.Define(stage, path)
    if any(v != 0.0 for v in world_pos):
        UsdGeom.Xformable(xform.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3f(*world_pos)
        )
    _apply_rigid_body(xform.GetPrim(), mass)
    return xform.GetPrim()


def _add_chassis_col(stage, parent_path):
    """차체 충돌 형상 (Box) — 물리 전용, 비가시."""
    hL, hW, hH = CHASSIS_L / 2.0, CHASSIS_W / 2.0, CHASSIS_H / 2.0
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/chassis_col")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(hL * 2.0, hW * 2.0, hH * 2.0)
    )
    _apply_collision(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()


def _add_knuckle_col(stage, parent_path):
    """너클 충돌 형상 (0.04 m 큐브) — 물리 전용, 비가시."""
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/knuckle_col")
    cube.GetSizeAttr().Set(0.04)
    _apply_collision(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()


def _add_wheel_col(stage, parent_path):
    """바퀴 충돌 형상 (Sphere) — 물리 전용, 비가시.

    PhysX 네이티브 Sphere 는 edge 가 전혀 없어 접지 안정성이 가장 높다.
    Cylinder/Capsule 의 평면 모서리 edge contact 로 인한 spin 자가증폭을 방지한다.
    """
    sph = UsdGeom.Sphere.Define(stage, f"{parent_path}/wheel_col")
    sph.GetRadiusAttr().Set(WHEEL_RADIUS)
    _apply_collision(sph.GetPrim())
    UsdGeom.Imageable(sph.GetPrim()).MakeInvisible()


def _add_lidar_sensor(stage, parent_path):
    """base_link 상단에 내장 RTX LiDAR Camera prim 추가."""
    cam = UsdGeom.Camera.Define(stage, f"{parent_path}/Lidar")
    prim = cam.GetPrim()

    if _RTX_LIDAR:
        IsaacSensorSchema.IsaacRtxLidarSensorAPI.Apply(prim)

    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, LIDAR_Z))

    sensor_type = prim.CreateAttribute("cameraSensorType", Sdf.ValueTypeNames.Token, False)
    sensor_type.Set("lidar")
    if not sensor_type.GetMetadata("allowedTokens"):
        sensor_type.SetMetadata("allowedTokens", ["camera", "radar", "lidar"])
    prim.CreateAttribute("sensorModelPluginName", Sdf.ValueTypeNames.String, False).Set(
        "omni.sensors.nv.lidar.lidar_core.plugin"
    )
    prim.CreateAttribute("sensorModelConfig", Sdf.ValueTypeNames.String, False).Set(
        "OS1_REV6_32ch10hz1024res"
    )

    cyl = UsdGeom.Cylinder.Define(stage, f"{parent_path}/Lidar/lidar_visual")
    cyl.GetRadiusAttr().Set(LIDAR_RADIUS)
    cyl.GetHeightAttr().Set(LIDAR_HEIGHT)
    UsdGeom.Gprim(cyl).GetDisplayColorAttr().Set([Gf.Vec3f(0.08, 0.08, 0.08)])


def _read_stl_binary(stl_path: str):
    """Binary STL → (Vt.Vec3fArray, face_counts, face_indices).

    STL은 삼각형마다 3개의 정점을 갖는다. 중복 정점 제거 없이 그대로 사용한다.
    """
    with open(stl_path, "rb") as f:
        data = f.read()
    n_tri = struct.unpack_from("<I", data, 80)[0]
    pts = []
    offset = 84
    for _ in range(n_tri):
        offset += 12  # normal 건너뜀
        v0 = struct.unpack_from("<3f", data, offset); offset += 12
        v1 = struct.unpack_from("<3f", data, offset); offset += 12
        v2 = struct.unpack_from("<3f", data, offset); offset += 12
        offset += 2   # attribute 건너뜀
        pts.append(Gf.Vec3f(*v0))
        pts.append(Gf.Vec3f(*v1))
        pts.append(Gf.Vec3f(*v2))
    vt_pts    = Vt.Vec3fArray(pts)
    vt_counts = Vt.IntArray([3] * n_tri)
    vt_idxs   = Vt.IntArray(list(range(n_tri * 3)))
    return vt_pts, vt_counts, vt_idxs


def _build_mesh_stage(usd_dir: str) -> str:
    """7개 링크의 STL을 읽어 hunter_se_v0_meshes.usdc(바이너리) 생성.

    각 링크 메시는 /Meshes/<link_name> 에 저장된다.
    바퀴 링크는 URDF joint rpy(±90°)에 맞춰 orient xform op을 추가한다.
    """
    mesh_path = os.path.join(usd_dir, "hunter_se_v0_meshes.usdc")
    stage = Usd.Stage.CreateNew(mesh_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Scope.Define(stage, "/Meshes")

    # (link_name, stl_file, color, orient_quat)
    # orient: URDF joint rpy 에서 바퀴가 Rx(±90°) 로 회전되어 있으므로 보정
    _LINKS = [
        ("base_link",           "base_link.STL",            _COLOR_BODY,  None),
        ("fr_steer_left_link",  "fr_steer_left_link.STL",   _COLOR_WHEEL, None),
        ("fr_left_link",        "fr_left_link.STL",          _COLOR_WHEEL, _Q_RX_POS),
        ("fr_steer_right_link", "fr_steer_right_link.STL",  _COLOR_WHEEL, None),
        ("fr_right_link",       "fr_right_link.STL",         _COLOR_WHEEL, _Q_RX_NEG),
        ("re_left_link",        "re_left_link.STL",          _COLOR_WHEEL, _Q_RX_POS),
        ("re_right_link",       "re_right_link.STL",         _COLOR_WHEEL, _Q_RX_NEG),
    ]

    for link_name, stl_file, color, orient in _LINKS:
        stl_path = os.path.join(_MESH_DIR, stl_file)
        if not os.path.exists(stl_path):
            print(f"[build_usd] STL 없음, 건너뜀: {stl_path}")
            continue
        print(f"[build_usd] STL 읽기: {stl_file} ...")
        pts, counts, idxs = _read_stl_binary(stl_path)

        # 링크별 루트는 Xform — 회전 보정을 여기에 적용
        # visual 참조 시 "def Xform visual → def Xform /Meshes/<link>" 로 타입이 일치해야
        # 메시 자식까지 렌더링됨. Xform 루트 아래에 Mesh 자식을 두는 것이 정석.
        xf_path = f"/Meshes/{link_name}"
        xf = UsdGeom.Xform.Define(stage, xf_path)
        if orient is not None:
            UsdGeom.Xformable(xf.GetPrim()).AddOrientOp().Set(orient)

        mesh = UsdGeom.Mesh.Define(stage, f"{xf_path}/mesh")
        mesh.GetPointsAttr().Set(pts)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(idxs)
        mesh.GetSubdivisionSchemeAttr().Set("none")
        UsdGeom.Gprim(mesh).GetDisplayColorAttr().Set([color])

    stage.GetRootLayer().Save()
    print(f"[build_usd] 메시 스테이지 저장: {mesh_path}")
    return mesh_path


def _add_stl_vis(stage, link_path: str, mesh_stage_path: str, link_name: str):
    """링크 아래에 /visual Xform을 추가하고 mesh USDC 의 해당 Xform prim을 참조.

    mesh USDC 구조: /Meshes/<link_name> (Xform) → /mesh (Mesh)
    여기서 visual prim도 Xform 이므로 타입이 일치 → 메시 자식까지 렌더링됨.
    """
    vis_xf = UsdGeom.Xform.Define(stage, f"{link_path}/visual")
    vis_xf.GetPrim().GetReferences().AddReference(
        assetPath=mesh_stage_path,
        primPath=Sdf.Path(f"/Meshes/{link_name}"),
    )


def _add_angular_drive(
    joint_prim,
    stiffness: float,
    damping: float,
    max_force: float,
):
    """PhysicsDriveAPI:angular 를 관절 prim에 추가한다.

    Isaac Lab의 ImplicitActuatorCfg 는 런타임에 stiffness/damping 을 덮어쓰지만,
    USD에 드라이브 API 자체가 없으면 position control 이 전혀 동작하지 않는다.
    원본 hunter_se Physics.usda 에서도 모든 관절에 PhysicsDriveAPI:angular 가
    미리 정의되어 있음.

    drive 종류:
        stiffness > 0 → position drive → targetPosition = 0 설정
        stiffness = 0 → velocity drive → targetVelocity = 0 설정 (원본 Physics.usda 방식)
    """
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive.GetTypeAttr().Set("force")
    drive.GetStiffnessAttr().Set(float(stiffness))
    drive.GetDampingAttr().Set(float(damping))
    drive.GetMaxForceAttr().Set(float(max_force))
    if stiffness > 0.0:
        drive.GetTargetPositionAttr().Set(0.0)   # position drive: 목표 각도 = 0
    else:
        drive.GetTargetVelocityAttr().Set(0.0)   # velocity drive: 목표 속도 = 0


def _make_revolute_joint(
    stage, joint_path,
    body0_path, body1_path,
    local_pos0, local_pos1,
    axis="Z",
    lower_deg=None, upper_deg=None,
    drive_stiffness: float = 0.0,
    drive_damping: float = 0.0,
    drive_max_force: float = 1e6,
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    joint.GetLocalRot0Attr().Set(_identity())
    joint.GetLocalRot1Attr().Set(_identity())
    joint.GetAxisAttr().Set(axis)
    if lower_deg is not None:
        joint.GetLowerLimitAttr().Set(float(lower_deg))
    if upper_deg is not None:
        joint.GetUpperLimitAttr().Set(float(upper_deg))
    if drive_stiffness > 0.0 or drive_damping > 0.0:
        _add_angular_drive(
            joint.GetPrim(),
            stiffness=drive_stiffness,
            damping=drive_damping,
            max_force=drive_max_force,
        )
    return joint


# ── 메인 빌드 함수 ─────────────────────────────────────────────────────────────

def build(output_path: str) -> str:
    """Hunter SE V0 USD articulation 파일 생성."""

    usd_dir = os.path.dirname(os.path.abspath(output_path))
    mesh_stage_path = _build_mesh_stage(usd_dir)

    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    ROOT = "/HunterSEV0"

    # ── 루트 Xform (ArticulationRootAPI) ────────────────────────────────────
    root_xf = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root_xf.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root_xf.GetPrim())
    if _PHYSX:
        PhysxSchema.PhysxArticulationAPI.Apply(root_xf.GetPrim())

    # ── 모든 링크를 루트의 직접 자식으로 배치 (플랫 계층) ────────────────────
    #    → RigidBody 중첩 없음, !resetXformStack! 불필요

    CHASSIS   = f"{ROOT}/base_link"
    FL_STEER  = f"{ROOT}/fr_steer_left_link"
    FL_WHEEL  = f"{ROOT}/fr_left_link"
    FR_STEER  = f"{ROOT}/fr_steer_right_link"
    FR_WHEEL  = f"{ROOT}/fr_right_link"
    RL_WHEEL  = f"{ROOT}/re_left_link"
    RR_WHEEL  = f"{ROOT}/re_right_link"

    # 차체
    _make_link(stage, CHASSIS, CHASSIS_MASS, world_pos=(0, 0, 0))
    _add_chassis_col(stage, CHASSIS)
    _add_stl_vis(stage, CHASSIS, mesh_stage_path, "base_link")

    # 전륜 좌 너클
    _make_link(stage, FL_STEER, KNUCKLE_MASS,
               world_pos=(FRONT_AX_X, FRONT_AX_Y, FRONT_AX_Z))
    _add_knuckle_col(stage, FL_STEER)
    _add_stl_vis(stage, FL_STEER, mesh_stage_path, "fr_steer_left_link")

    # 전륜 좌 바퀴 (너클과 동일 위치 = 조향 피벗 = 바퀴 중심)
    _make_link(stage, FL_WHEEL, WHEEL_MASS,
               world_pos=(FRONT_AX_X, FRONT_AX_Y, FRONT_AX_Z))
    _add_wheel_col(stage, FL_WHEEL)
    _add_stl_vis(stage, FL_WHEEL, mesh_stage_path, "fr_left_link")

    # 전륜 우 너클
    _make_link(stage, FR_STEER, KNUCKLE_MASS,
               world_pos=(FRONT_AX_X, -FRONT_AX_Y, FRONT_AX_Z))
    _add_knuckle_col(stage, FR_STEER)
    _add_stl_vis(stage, FR_STEER, mesh_stage_path, "fr_steer_right_link")

    # 전륜 우 바퀴
    _make_link(stage, FR_WHEEL, WHEEL_MASS,
               world_pos=(FRONT_AX_X, -FRONT_AX_Y, FRONT_AX_Z))
    _add_wheel_col(stage, FR_WHEEL)
    _add_stl_vis(stage, FR_WHEEL, mesh_stage_path, "fr_right_link")

    # 후륜 좌 바퀴
    _make_link(stage, RL_WHEEL, WHEEL_MASS,
               world_pos=(REAR_AX_X, REAR_AX_Y, REAR_AX_Z))
    _add_wheel_col(stage, RL_WHEEL)
    _add_stl_vis(stage, RL_WHEEL, mesh_stage_path, "re_left_link")

    # 후륜 우 바퀴
    _make_link(stage, RR_WHEEL, WHEEL_MASS,
               world_pos=(REAR_AX_X, -REAR_AX_Y, REAR_AX_Z))
    _add_wheel_col(stage, RR_WHEEL)
    _add_stl_vis(stage, RR_WHEEL, mesh_stage_path, "re_right_link")

    # ── 조인트 ────────────────────────────────────────────────────────────────
    PHYS = f"{ROOT}/Physics"
    UsdGeom.Scope.Define(stage, PHYS)

    # 전륜 조향: base_link → fr_steer_left  (axis=Z, 수직축 회전)
    # drive_stiffness=1e7: 원본 Physics.usda 값과 동일.
    # Isaac Lab ImplicitActuatorCfg 가 런타임에 500/50 으로 덮어쓴다.
    # USD 에 PhysicsDriveAPI:angular 가 없으면 position 제어 자체가 불가.
    _make_revolute_joint(
        stage, f"{PHYS}/fr_steer_left_joint",
        CHASSIS, FL_STEER,
        local_pos0=(FRONT_AX_X,  FRONT_AX_Y, FRONT_AX_Z),
        local_pos1=(0, 0, 0),
        axis="Z",
        lower_deg=-MAX_STEER_DEG, upper_deg=MAX_STEER_DEG,
        drive_stiffness=1e7, drive_damping=1e5, drive_max_force=6000.0,
    )
    # 전륜 조향: base_link → fr_steer_right  (axis=Z)
    _make_revolute_joint(
        stage, f"{PHYS}/fr_steer_right_joint",
        CHASSIS, FR_STEER,
        local_pos0=(FRONT_AX_X, -FRONT_AX_Y, FRONT_AX_Z),
        local_pos1=(0, 0, 0),
        axis="Z",
        lower_deg=-MAX_STEER_DEG, upper_deg=MAX_STEER_DEG,
        drive_stiffness=1e7, drive_damping=1e5, drive_max_force=6000.0,
    )
    # 전륜 회전: fr_steer_left → fr_left  (axis=Y, 횡방향)
    # ImplicitActuatorCfg (front_wheels, damping=0.5) — 드라이브 API 필요
    # 원본 Physics.usda: stiffness=0, damping=0.5
    _make_revolute_joint(
        stage, f"{PHYS}/fr_left_joint",
        FL_STEER, FL_WHEEL,
        local_pos0=(0, 0, 0),
        local_pos1=(0, 0, 0),
        axis="Y",
        drive_stiffness=0.0, drive_damping=0.5, drive_max_force=1e6,
    )
    # 전륜 회전: fr_steer_right → fr_right  (axis=Y)
    _make_revolute_joint(
        stage, f"{PHYS}/fr_right_joint",
        FR_STEER, FR_WHEEL,
        local_pos0=(0, 0, 0),
        local_pos1=(0, 0, 0),
        axis="Y",
        drive_stiffness=0.0, drive_damping=0.5, drive_max_force=1e6,
    )
    # 후륜 구동: base_link → re_left  (axis=Y)
    # DCMotorCfg 가 런타임에 제어하지만, 원본과 동일하게 드라이브 API 정의
    # 원본 Physics.usda: stiffness=0, damping=17453, targetVelocity=0
    _make_revolute_joint(
        stage, f"{PHYS}/re_left_joint",
        CHASSIS, RL_WHEEL,
        local_pos0=(REAR_AX_X,  REAR_AX_Y, REAR_AX_Z),
        local_pos1=(0, 0, 0),
        axis="Y",
        drive_stiffness=0.0, drive_damping=17453.0, drive_max_force=1e6,
    )
    # 후륜 구동: base_link → re_right  (axis=Y)
    _make_revolute_joint(
        stage, f"{PHYS}/re_right_joint",
        CHASSIS, RR_WHEEL,
        local_pos0=(REAR_AX_X, -REAR_AX_Y, REAR_AX_Z),
        local_pos1=(0, 0, 0),
        axis="Y",
        drive_stiffness=0.0, drive_damping=17453.0, drive_max_force=1e6,
    )

    stage.GetRootLayer().Save()
    print(f"[build_usd] 저장 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    build(os.path.join(_dir, "hunter_se_v0.usda"))
