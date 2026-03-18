# ros2 — 실로봇 배포 (Phase 4 예정)

학습된 정책을 실제 Hunter SE 로봇에 배포하기 위한 ROS2 패키지를 담을 디렉터리입니다. 현재는 디렉터리 구조만 준비되어 있으며 Phase 4에서 구현됩니다.

---

## 예정 구조

```
ros2/
├── autodrive_interfaces/       # 커스텀 ROS2 인터페이스 (srv, action 정의)
└── isaaclab_ros2_bridge/       # 실로봇 배포 브리지 노드
```

---

## autodrive_interfaces (예정)

Hunter SE 자율주행에 필요한 커스텀 ROS2 메시지, 서비스, 액션 타입을 정의합니다.

예상 내용:
- 목표 지점 전달 서비스
- 정책 실행 상태 토픽
- 에피소드 결과 액션

---

## isaaclab_ros2_bridge (예정)

학습된 정책을 실로봇에서 실행하는 ROS2 노드입니다.

**예상 파이프라인:**

```
/scan (LaserScan)
    └─→ 80-sector 상태 변환 (LiDAR 전처리)
            └─→ ONNX/TorchScript 정책 추론
                    └─→ /cmd_vel (Twist) 발행
```

**예상 노드 구성:**
- `lidar_preprocessor`: `/scan` → 80-sector 정규화 관측 변환
- `policy_runner`: 관측 → 정책 추론 → 행동 출력
- `cmd_vel_publisher`: 행동 → `/cmd_vel` Twist 메시지 변환

---

## 빌드 방법 (예정)

```bash
# ros2 패키지를 워크스페이스에 심볼릭 링크
ln -s /workspace/hunter_autodrive/ros2/autodrive_interfaces \
      /robot_isaac/ros2_ws/src/
ln -s /workspace/hunter_autodrive/ros2/isaaclab_ros2_bridge \
      /robot_isaac/ros2_ws/src/

# 빌드
cd /robot_isaac/ros2_ws
colcon build --packages-select autodrive_interfaces isaaclab_ros2_bridge

# 실행
ros2 launch isaaclab_ros2_bridge deploy_hunter.launch.py \
    policy_path:=/path/to/policy.onnx
```

---

## 전제 조건

1. Phase 3-E까지 학습된 체크포인트 존재
2. `scripts/autodrive/deploy/export_policy.py`로 ONNX 내보내기 완료
3. 실로봇의 LiDAR 토픽이 `/scan` (LaserScan, 360° 스캔) 으로 발행 중
4. Sim-to-Real 갭 분석 및 도메인 랜덤화 적용
