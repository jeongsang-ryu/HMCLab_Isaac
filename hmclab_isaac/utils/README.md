# utils/

카테고리 공용 유틸. 현재:

- `ros2.py` — `attach_ros2_graph(robot_prim_path, topics)`. Python-driven OmniGraph 래퍼. 지금은 스텁, M7에서 채움.

**원칙**: 여기 들어가는 것은 racing·offroad 양쪽이 다 쓸 수 있는 것만. 카테고리 전용 유틸은 `robots/racing/_utils/` 같은 하위 폴더로 분리.
