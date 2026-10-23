from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

q_des = 0.8


def make_model():
    # 每组实验都重新加载一份 model,避免上一组仿真的状态或参数污染下一组。
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("pendulum.xml")))

    # 这里统一使用 motor 通道手写 PD。
    # 所以要关掉 XML 里同时存在的 position 执行器,否则会叠加两份控制力矩。
    pos_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_act")
    model.actuator_gainprm[pos_act_id, :] = 0.0  # 清掉 position 执行器的 kp
    model.actuator_biasprm[pos_act_id, :] = 0.0  # 清掉 position 执行器的 kv/偏置项
    return model


def run_pd(Kp, Kd, disturbance=False):
    model = make_model()
    data = mujoco.MjData(model)
    ts, qs = [], []  # 只记录时间和角度,用于画响应曲线

    while data.time < 8.0:
        q = data.qpos[0]   # 当前关节角度
        dq = data.qvel[0]  # 当前关节角速度

        # 手写 PD:目标速度设为 0,所以 D 项是 Kd * (0 - dq)。
        tau = Kp * (q_des - q) + Kd * (0.0 - dq)

        # 写入 motor 执行器的 ctrl[1];clip 对应 XML 里的 ctrlrange="-5 5"。
        data.ctrl[1] = np.clip(tau, -5.0, 5.0)

        # 实验 2 使用:3 秒后施加恒定外扰,观察 PD 是否能消掉稳态误差。
        data.qfrc_applied[0] = 2.0 if disturbance and data.time > 3.0 else 0.0

        mujoco.mj_step(model, data)
        ts.append(data.time)
        qs.append(data.qpos[0])

    return np.array(ts), np.array(qs)


plt.figure(figsize=(8, 4))
# 实验 1:固定 Kp,只改变 Kd,看欠阻尼 / 临界附近 / 过阻尼三种响应。
for Kd, label in [(0.5, "A: Kd=0.5"), (12.0, "B: Kd=12"), (40.0, "C: Kd=40")]:
    t, q = run_pd(Kp=40.0, Kd=Kd)
    plt.plot(t, q, label=label)
plt.axhline(q_des, linestyle="--", color="black", label="target")
plt.xlabel("time (s)")
plt.ylabel("q (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pd_kd_sweep.png", dpi=200)

plt.figure(figsize=(8, 4))
# 实验 2:加入恒定外扰,对比纯 P 和 PD 的稳态误差。
for Kd, label in [(0.0, "P only"), (1.0, "PD")]:
    t, q = run_pd(Kp=20.0, Kd=Kd, disturbance=True)
    plt.plot(t, q, label=label)
plt.axhline(q_des, linestyle="--", color="black", label="target")
plt.axvline(3.0, linestyle=":", color="gray", label="disturbance")
plt.xlabel("time (s)")
plt.ylabel("q (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pd_disturbance_error.png", dpi=200)