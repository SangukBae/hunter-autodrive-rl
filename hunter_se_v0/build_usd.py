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
    hunter_se/Payload/Physics.usda   — 관절 위치·방향
    hunter_se/Payload/Geometry.usda  — 링크 위치·방향
    hunter_se/Payload/Physics.usda   — 질량 분배

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

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema
    _PHYSX = True
except ImportError:
    _PHYSX = False

# ── 기구 상수 (hunter_se URDF Physics.usda 정확한 값) ─────────────────────────
FRONT_AX_X   =  0.34058   # 전륜 관절 X [m]
FRONT_AX_Y   =  0.24619   # 전륜 조향 피벗 ±Y [m]
FRONT_AX_Z   = -0.1535    # 전륜 관절 Z [m]
REAR_AX_X    = -0.2078    # 후륜 관절 X [m]
REAR_AX_Y    =  0.252     # 후륜 허브 ±Y [m]
REAR_AX_Z    = -0.158     # 후륜 관절 Z [m]

WHEEL_RADIUS =  0.129     # 바퀴 반지름 [m]
WHEEL_WIDTH  =  0.080     # 바퀴 폭 [m]
MAX_STEER_DEG = 22.0      # 최대 조향각 [deg]

# 차체 외형 (PDF 기준)
CHASSIS_L    =  0.817
CHASSIS_W    =  0.580
CHASSIS_H    =  0.120

# 질량 (URDF Physics.usda 값)
CHASSIS_MASS = 29.39
KNUCKLE_MASS =  3.149
WHEEL_MASS   =  3.149

# hunter_se 원본 GeometryLibrary 참조 경로 (v0 USD 위치 기준 상대 경로)
GEOM_LIB = "../hunter_se/Payload/GeometryLibrary.usdc"

# 바퀴 시각화 오리엔트 (w, x, y, z) — Geometry.usda resetXformStack 값 기반
# 계산: 각 링크 Xform orient × 내부 Mesh orient (쿼터니언 곱)
# fr_left_link : (0.7071, 0.7071, 0, 0) × (0, 1, 0, 0) × (0, 1, 0, 0)[X flip]
#              = (-0.7071, 0.7071, 0, 0) × (0, 1, 0, 0) = (-0.7071, -0.7071, 0, 0)
# fr_right_link: (-0.7071, 0.7071, 0, 0) × (1, 0, 0, 0) = (-0.7071, 0.7071, 0, 0)
# re_left_link : (0.7071, 0.7071, 0, 0) × (1, 0, 0, 0) = ( 0.7071, 0.7071, 0, 0)
# re_right_link: (-0.7071, 0.7071, 0, 0) × (1, 0, 0, 0) = (-0.7071, 0.7071, 0, 0)
_Q_FL = (-0.70710546, -0.7071081,  0.0, 0.0)   # 180° X flip 추가 적용
_Q_FR = (-0.70710546,  0.7071081,  0.0, 0.0)
_Q_RL = ( 0.70710546,  0.7071081,  0.0, 0.0)
_Q_RR = (-0.70710546,  0.7071081,  0.0, 0.0)


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


def _add_vis_ref(stage, parent_path, name, lib_prim_path, orient_wxyz=None):
    """hunter_se GeometryLibrary 메시 참조를 시각화 전용으로 추가.

    물리·충돌 형상과 무관한 순수 시각화 레이어.
    orient_wxyz: (w, x, y, z) — 원본 Geometry.usda의 resetXformStack 절대 오리엔트.
    """
    xf = UsdGeom.Xform.Define(stage, f"{parent_path}/{name}")
    if orient_wxyz is not None:
        UsdGeom.Xformable(xf.GetPrim()).AddOrientOp(
            UsdGeom.XformOp.PrecisionFloat
        ).Set(Gf.Quatf(*orient_wxyz))
    mesh = stage.DefinePrim(f"{parent_path}/{name}/mesh")
    mesh.GetReferences().AddReference(
        assetPath=GEOM_LIB,
        primPath=Sdf.Path(lib_prim_path),
    )
    # 원본 Geometry.usda 패턴: 라이브러리 메시의 xformOp를 정적 identity로 덮어쓴다.
    # GeometryLibrary.usdc 내부에 시간 기반 또는 비-identity xformOp가 있을 수 있으며,
    # 이를 덮어쓰지 않으면 re_left_link 등이 정지 상태에서도 회전하는 증상이 발생한다.
    xformable = UsdGeom.Xformable(mesh)
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _add_chassis_col(stage, parent_path):
    """차체 충돌 형상 (Box, 렌더링 비활성)."""
    hL, hW, hH = CHASSIS_L / 2.0, CHASSIS_W / 2.0, CHASSIS_H / 2.0
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/chassis_col")
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(hL * 2.0, hW * 2.0, hH * 2.0)
    )
    _apply_collision(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()


def _add_knuckle_col(stage, parent_path):
    """너클 충돌 형상 (0.04 m 큐브, 렌더링 비활성)."""
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/knuckle_col")
    cube.GetSizeAttr().Set(0.04)
    _apply_collision(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()


def _add_wheel_col(stage, parent_path):
    """바퀴 충돌 형상 (Cylinder, axis=Y, 렌더링 비활성)."""
    cyl = UsdGeom.Cylinder.Define(stage, f"{parent_path}/wheel_col")
    cyl.GetRadiusAttr().Set(WHEEL_RADIUS)
    cyl.GetHeightAttr().Set(WHEEL_WIDTH)
    cyl.GetAxisAttr().Set("Y")
    _apply_collision(cyl.GetPrim())
    UsdGeom.Imageable(cyl.GetPrim()).MakeInvisible()


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
    _add_vis_ref(stage, CHASSIS, "chassis_vis", "/Geometry/base_link")

    # 전륜 좌 너클
    _make_link(stage, FL_STEER, KNUCKLE_MASS,
               world_pos=(FRONT_AX_X, FRONT_AX_Y, FRONT_AX_Z))
    _add_knuckle_col(stage, FL_STEER)

    # 전륜 좌 바퀴 (너클과 동일 위치 = 조향 피벗 = 바퀴 중심)
    _make_link(stage, FL_WHEEL, WHEEL_MASS,
               world_pos=(FRONT_AX_X, FRONT_AX_Y, FRONT_AX_Z))
    _add_wheel_col(stage, FL_WHEEL)
    _add_vis_ref(stage, FL_WHEEL, "wheel_vis", "/Geometry/fr_right_link", orient_wxyz=_Q_FL)

    # 전륜 우 너클
    _make_link(stage, FR_STEER, KNUCKLE_MASS,
               world_pos=(FRONT_AX_X, -FRONT_AX_Y, FRONT_AX_Z))
    _add_knuckle_col(stage, FR_STEER)

    # 전륜 우 바퀴
    _make_link(stage, FR_WHEEL, WHEEL_MASS,
               world_pos=(FRONT_AX_X, -FRONT_AX_Y, FRONT_AX_Z))
    _add_wheel_col(stage, FR_WHEEL)
    _add_vis_ref(stage, FR_WHEEL, "wheel_vis", "/Geometry/fr_right_link", orient_wxyz=_Q_FR)

    # 후륜 좌 바퀴
    _make_link(stage, RL_WHEEL, WHEEL_MASS,
               world_pos=(REAR_AX_X, REAR_AX_Y, REAR_AX_Z))
    _add_wheel_col(stage, RL_WHEEL)
    _add_vis_ref(stage, RL_WHEEL, "wheel_vis", "/Geometry/re_left_link", orient_wxyz=_Q_RL)

    # 후륜 우 바퀴
    _make_link(stage, RR_WHEEL, WHEEL_MASS,
               world_pos=(REAR_AX_X, -REAR_AX_Y, REAR_AX_Z))
    _add_wheel_col(stage, RR_WHEEL)
    _add_vis_ref(stage, RR_WHEEL, "wheel_vis", "/Geometry/re_left_link", orient_wxyz=_Q_RR)

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
