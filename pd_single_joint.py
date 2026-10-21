from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("pendulum.xml")))
data = mujoco.MjData(model)

q_des = 0.8           # 目标角度(rad),约 45.8°; position 执行器和手写 PD 都跟踪这个目标
Kp, Kd = 20.0, 1.0    # 手写 PD 的比例/微分增益;刻意与 XML 里的 kp/kv 一致,方便对照

use_motor = True      # True: 写 ctrl[1],用 motor 手搓 PD; False: 写 ctrl[0],用 position 执行器

if use_motor:
    # MJCF 里同时定义了 pos_act 和 tor_act。
    # 如果不关掉 pos_act,它会继续按 ctrl[0] 输出内置 PD 力矩,
    # 这样就变成"内置 PD + 手写 PD"叠加,无法单独观察 motor 写法。
    pos_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_act")
    model.actuator_gainprm[pos_act_id, :] = 0.0  # 清掉 position 执行器的 kp 增益
    model.actuator_biasprm[pos_act_id, :] = 0.0  # 清掉 position 执行器的 kv/偏置项

log = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 8.0:
        step_start = time.time()

        q = data.qpos[0]     # hinge 当前角度 q,对应 XML 里唯一的 joint
        dq = data.qvel[0]    # hinge 当前角速度 dq/dt

        if use_motor:
            # motor 模式:自己计算 tau = Kp(q_des - q) + Kd(0 - dq)
            # 目标速度设为 0,表示只要求最终停在目标角度,不要求持续运动。
            tau = Kp * (q_des - q) + Kd * (0.0 - dq)
            data.ctrl[1] = np.clip(tau, -5.0, 5.0)   # 写入 tor_act;clip 对应 XML 的 ctrlrange="-5 5"
        else:
            # position 模式:只给目标角度,MuJoCo 在内部用 kp/kv 自动算出关节力矩。
            data.ctrl[0] = q_des                      # 写入 pos_act;范围受 ctrlrange="-3.14 3.14" 限制

        mujoco.mj_step(model, data)
        viewer.sync()
        log.append((data.time, q, dq, data.ctrl.copy()))

        # 让仿真按 wall-clock 走,viewer 看起来才是实时的
        dt_left = model.opt.timestep - (time.time() - step_start)
        if dt_left > 0:
            time.sleep(dt_left)