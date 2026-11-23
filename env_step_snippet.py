"""env_step_snippet.py -- 拐杖机器人 step 接口（原生 mujoco Python API, MuJoCo 3.4）

与 crutch_robot.xml 配对使用；人体与拐杖共用同一 MjModel / MjData。

集成到 MyoFullBody 时只需改 XML 两处，其余由本文件承担：
  1. grip_l 的 body2 -> "radius_l"，grip_r 的 body2 -> "radius_r"
  2. 删除演示用 hand_stub_l / hand_stub_r 两个刚体
  3. 拐杖 actuator 保持在人体 actuator 之后（肌肉动作顺序与维度不变）
  4. 用合并后的 model/data 构造 CrutchInterface，并在回合初始化时调用 align_grip()

传感器口径（已用已知竖直载荷标定，见 __main__ 自检）：
  grip_force  挂在 handle 刚体的 grip_sensor site 上，读数为 site 坐标系
              三分量，量值 = 该刚体承受的合力（含把手自重）。
              世界竖直分量 = (R_handle @ f_raw)[2]；去皮基准 tare 在无连接、
              静止时取一次，故
                  support_up = +((R_handle @ f_raw)[2] - tare[2])
              为「拐杖对手向上为正」的竖直支撑力。
              残差含把手自身惯性项 m_handle * a_handle（静态时 < 0.5 N），
              不做逐帧惯量补偿；需要高频力控时用同式减去该项。
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# 缺省参数：集中在此，均为可修改的仿真初值
# --------------------------------------------------------------------------
DEFAULTS = {
    "xml": "crutch_robot.xml",
    # 几何（与 crutch_robot.xml 对齐，validate() 会核对）
    "chassis_half_size": (0.195, 0.145, 0.125),
    "chassis_mass": 15.5,
    "rod_length": 1.00,
    "clearance": 0.05,
    "hand_height_ref": 0.90,
    # 随行控制
    "forward_offset": 0.55,
    "lateral_offset": 0.22,
    "k_ff": 1.00,
    "k_pos": 2.50,
    "k_vel": 0.85,
    "v_max": 1.20,
    "cmd_dv_coef": 0.50,
    "frame_skip": 5,
    "heading_speed_floor": 0.05,
    # 握持与辅助
    "grip_dist_tol": 0.05,
    "hand_body": {"left": "hand_stub_l", "right": "hand_stub_r"},
    "hand_site": {"left": "hand_site_l", "right": "hand_site_r"},
    "free_side": "left",
    "muscle_ctrl_ids": None,
}

ASSIST_MODES = ("none", "ipsilateral", "contralateral")


def _id(m, objtype, name):
    i = mujoco.mj_name2id(m, objtype, name)
    if i < 0:
        raise KeyError(f"model has no {objtype!r} named {name!r}")
    return i


class CrutchInterface:
    """底盘随行控制 + 握持连接管理 + 力传感读取。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: dict | None = None):
        self.m, self.d = model, data
        # 调用方可能只给了 MjData(model)，此时派生量（site_xpos/xmat/sensordata）尚未有效
        mujoco.mj_forward(model, data)
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.control_dt = float(model.opt.timestep) * int(self.cfg["frame_skip"])

        self.body_chassis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.body_handle = _id(model, mujoco.mjtObj.mjOBJ_BODY, "handle")
        self.site_grip = _id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_point")
        self.site_chassis = _id(model, mujoco.mjtObj.mjOBJ_SITE, "chassis_center")
        self.jnt_x = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
        self.jnt_y = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")
        self.dof_x = int(model.jnt_dofadr[self.jnt_x])
        self.dof_y = int(model.jnt_dofadr[self.jnt_y])
        self.act = {n: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("crutch_x", "crutch_y")}
        self.eq = {s: _id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{s[0]}") for s in ("left", "right")}
        self.sens = {n: _id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                     for n in ("grip_force", "grip_torque", "chassis_pos", "chassis_vel")}
        self.body_hand, self.site_hand = {}, {}
        for s in ("left", "right"):
            self.body_hand[s] = _id(model, mujoco.mjtObj.mjOBJ_BODY, self.cfg["hand_body"][s])
            self.site_hand[s] = _id(model, mujoco.mjtObj.mjOBJ_SITE, self.cfg["hand_site"][s])

        ids = [i for i in range(model.nu) if i not in self.act.values()]
        self.muscle_ids = np.array(self.cfg["muscle_ctrl_ids"] if self.cfg["muscle_ctrl_ids"] is not None else ids, dtype=int)
        self.tare = np.zeros(3)
        self.tare_set = False
        self._v_prev = np.zeros(2)
        self._v_meas_prev = np.zeros(2)
        self._heading = np.array([1.0, 0.0])
        self.mode, self.affected_side, self.active_side = "none", "left", None
        self.last: dict = {}

    # ---------------- 读取 ----------------
    def _sensor(self, name, dim):
        adr = int(self.m.sensor_adr[self.sens[name]])
        return np.array(self.d.sensordata[adr:adr + dim], dtype=float)

    def read(self) -> dict:
        f_raw = self._sensor("grip_force", 3)
        t_raw = self._sensor("grip_torque", 3)
        rot = self.d.xmat[self.body_handle].reshape(3, 3)
        f_world = rot @ f_raw
        support_up = float(f_world[2] - self.tare[2]) if self.tare_set else float("nan")
        gap = {s: float(np.linalg.norm(self.d.site_xpos[self.site_grip] - self.d.site_xpos[self.site_hand[s]]))
               for s in ("left", "right")}
        return {
            "grip_force_raw": f_raw,
            "grip_torque_raw": t_raw,
            "grip_force_world": f_world,
            "support_up_N": support_up,
            "grip_gap_m": gap,
            "eq_active": {s: bool(self.d.eq_active[self.eq[s]]) for s in ("left", "right")},
        }

    def tare_now(self) -> np.ndarray:
        """无连接、静止时取一次去皮基准；连接前必须调用。"""
        rot = self.d.xmat[self.body_handle].reshape(3, 3)
        self.tare = rot @ self._sensor("grip_force", 3)
        self.tare_set = True
        return self.tare

    # ---------------- 握持 ----------------
    def align_grip(self, side: str | None = None) -> dict:
        """回合初始化时对齐握点：平移底盘使握点落在该手 site，并重算 anchor2。

        只能在没有力学历史时调用（不覆盖仿真步进中的 qpos/qvel）。
        返回剩余偏差；竖直方向不可由底盘修正。
        """
        side = side or self.active_side or self.affected_side
        if self.d.eq_active[self.eq[side]]:
            raise RuntimeError(f"align_grip({side}) called while the connection is active; "
                               "deactivate with set_assistance('none') first")
        p_hand = self.d.site_xpos[self.site_hand[side]].copy()
        grip_local = self.d.xmat[self.body_chassis].reshape(3, 3).T @ (
            self.d.site_xpos[self.site_grip] - self.d.xpos[self.body_chassis])
        qadr = int(self.m.jnt_qposadr[self.jnt_x])
        self.d.qpos[qadr:qadr + 2] = p_hand[:2] - grip_local[:2]
        mujoco.mj_forward(self.m, self.d)

        i = self.eq[side]
        self.m.eq_data[i][0:3] = self.d.xmat[self.body_handle].reshape(3, 3).T @ (
            self.d.site_xpos[self.site_grip] - self.d.xpos[self.body_handle])
        self.m.eq_data[i][3:6] = self.d.xmat[self.body_hand[side]].reshape(3, 3).T @ (
            self.d.site_xpos[self.site_grip] - self.d.xpos[self.body_hand[side]])
        mujoco.mj_forward(self.m, self.d)
        gap = float(np.linalg.norm(self.d.site_xpos[self.site_grip] - p_hand))
        return {"side": side, "gap_m": gap, "aligned": gap <= self.cfg["grip_dist_tol"]}

    def set_assistance(self, mode: str, affected_side: str = "left", force: bool = False) -> dict:
        """mode: none / contralateral(健侧持杖) / ipsilateral(患侧持杖)。"""
        if mode not in ASSIST_MODES:
            raise ValueError(f"mode must be one of {list(ASSIST_MODES)}, got {mode!r}")
        if affected_side not in ("left", "right"):
            raise ValueError(f"affected_side must be left/right, got {affected_side!r}")
        target = None
        if mode == "ipsilateral":
            target = affected_side
        elif mode == "contralateral":
            target = "right" if affected_side == "left" else "left"

        gap_tol = float(self.cfg["grip_dist_tol"])
        if target is not None and not force:
            gap = float(np.linalg.norm(self.d.site_xpos[self.site_grip] - self.d.site_xpos[self.site_hand[target]]))
            if gap > gap_tol:
                for s in ("left", "right"):
                    self.d.eq_active[self.eq[s]] = 0
                self.active_side = None
                self.mode, self.affected_side = mode, affected_side
                return {"ok": False, "reason": "grip_offset_too_large", "gap_m": gap,
                        "active_side": None, "eq_active": {s: False for s in ("left", "right")}}

        for s in ("left", "right"):
            self.d.eq_active[self.eq[s]] = 1 if s == target else 0
        self.mode, self.affected_side, self.active_side = mode, affected_side, target
        return {"ok": True, "reason": "activated" if target else "no_assistance",
                "gap_m": (float(np.linalg.norm(self.d.site_xpos[self.site_grip] - self.d.site_xpos[self.site_hand[target]]))
                          if target else float("nan")),
                "active_side": target, "eq_active": {s: target == s for s in ("left", "right")}}

    # ---------------- 步进 ----------------
    def _plan(self, human_pos, human_vel) -> tuple[np.ndarray, dict]:
        v_h = np.asarray(human_vel, dtype=float)[:2]
        p_h = np.asarray(human_pos, dtype=float)[:2]
        sp = float(np.linalg.norm(v_h))
        if sp > self.cfg["heading_speed_floor"]:
            self._heading = v_h / sp
        fwd = self._heading
        left = np.array([-fwd[1], fwd[0]])
        side = self.active_side or self.cfg["free_side"]
        sign = 1.0 if side == "left" else -1.0
        target = p_h + fwd * self.cfg["forward_offset"] + left * self.cfg["lateral_offset"] * sign

        p_c = self._sensor("chassis_pos", 3)[:2]
        v_c = np.array([self.d.qvel[self.dof_x], self.d.qvel[self.dof_y]])
        v_des = (self.cfg["k_ff"] * v_h
                 + self.cfg["k_pos"] * (target - p_c)
                 + self.cfg["k_vel"] * (v_h - v_c))
        n = float(np.linalg.norm(v_des))
        speed_limited = n > self.cfg["v_max"]
        if speed_limited:
            v_des *= self.cfg["v_max"] / n

        dv = v_des - self._v_prev
        nd = float(np.linalg.norm(dv))
        dv_limit = self.cfg["cmd_dv_coef"] * self.control_dt
        dv_limited = nd > dv_limit
        if dv_limited:
            dv *= dv_limit / nd
        v_cmd = self._v_prev + dv
        info = {"target": target, "v_des": v_des, "dv_norm": float(np.linalg.norm(dv)),
                "dv_limit": dv_limit, "dv_limited": dv_limited, "speed_limited": speed_limited}
        return v_cmd, info

    def step(self, human_pos, human_vel, muscle_act) -> dict:
        """推进一个控制步：写入底盘速度伺服与人体肌肉动作，返回诊断量。"""
        u = np.asarray(muscle_act, dtype=float).ravel()
        if u.size != self.muscle_ids.size:
            raise ValueError(f"muscle action size {u.size} != model slots {self.muscle_ids.size}")

        v_cmd, info = self._plan(human_pos, human_vel)
        self.d.ctrl[self.act["crutch_x"]] = v_cmd[0]
        self.d.ctrl[self.act["crutch_y"]] = v_cmd[1]
        self.d.ctrl[self.muscle_ids] = u
        for _ in range(int(self.cfg["frame_skip"])):
            mujoco.mj_step(self.m, self.d)
        # mj_step 在积分前的 forward 阶段算传感器，这里补一次 forward，
        # 使读出的力/速度与控制步结束时刻的 t 对齐
        mujoco.mj_forward(self.m, self.d)

        v_meas = np.array([self.d.qvel[self.dof_x], self.d.qvel[self.dof_y]])
        a_meas = (v_meas - self._v_meas_prev) / self.control_dt
        self._v_prev = v_cmd
        self._v_meas_prev = v_meas

        forces = np.array([self.d.actuator_force[self.act["crutch_x"]], self.d.actuator_force[self.act["crutch_y"]]])
        fr = np.abs(self.m.actuator_forcerange[[self.act["crutch_x"], self.act["crutch_y"]], 1])
        out = self.read()
        out.update({
            "t": float(self.d.time),
            "v_cmd": v_cmd, "v_meas": v_meas, "a_meas": a_meas,
            "actuator_force": forces, "force_saturated": bool(np.any(np.abs(forces) >= fr - 1e-9)),
            **{k: v for k, v in info.items()},
        })
        self.last = out
        return out

    # ---------------- 验收辅助 ----------------
    def validate(self) -> dict:
        m = self.m
        gz = float(np.abs(m.opt.gravity[2]))
        box = _id(m, mujoco.mjtObj.mjOBJ_GEOM, "chassis_box")
        half_z = float(m.geom_size[box][2])
        handle_z = float(self.d.xpos[self.body_handle][2])
        rods = handle_z - float(self.d.xpos[self.body_chassis][2]) - half_z
        demo_hand_z = float(self.d.site_xpos[self.site_hand["left"]][2])
        ref_z = float(self.cfg["hand_height_ref"])
        rod_for_ref = ref_z - self.cfg["clearance"] - 2.0 * half_z
        checks = {
            "sensor_dims": {n: int(m.sensor_dim[self.sens[n]]) for n in ("grip_force", "grip_torque", "chassis_pos", "chassis_vel")},
            "sensor_dims_ok": bool(np.all([m.sensor_dim[self.sens["grip_force"]] == 3, m.sensor_dim[self.sens["grip_torque"]] == 3,
                                          m.sensor_dim[self.sens["chassis_pos"]] == 3, m.sensor_dim[self.sens["chassis_vel"]] == 3])),
            "chassis_box_size": [round(float(v), 4) for v in m.geom_size[box][:3]],
            "chassis_mass_kg": round(float(m.body_mass[self.body_chassis]), 4),
            "mass_ok": abs(float(m.body_mass[self.body_chassis]) - self.cfg["chassis_mass"]) < 0.05,
            "rod_length_measured": round(rods, 4),
            "rod_length_ok": abs(rods - self.cfg["rod_length"]) < 1e-3,
            "handle_height_m": round(handle_z, 4),
            "handle_height_formula": round(self.cfg["clearance"] + 2.0 * half_z + self.cfg["rod_length"], 4),
            "demo_hand_height_m": round(demo_hand_z, 4),
            "demo_handle_minus_hand_m": round(handle_z - demo_hand_z, 4),
            "hand_height_ref_m": round(ref_z, 4),
            "rod_length_for_hand_ref_m": round(rod_for_ref, 4),
            "handle_over_hand_ref_m": round(handle_z - ref_z, 4),
            "crutch_mass_kg": round(float(m.body_mass[self.body_chassis] + m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rod")]
                                          + m.body_mass[self.body_handle]), 3),
            "model_total_weight_N": round(float(m.body_mass.sum()) * gz, 2),
            "nq_nv_nu": [m.nq, m.nv, m.nu],
            "eq_inactive_at_start": bool(not np.any(self.d.eq_active)),
        }
        checks["ok"] = bool(checks["sensor_dims_ok"] and checks["mass_ok"] and checks["rod_length_ok"])
        return checks


def load(xml: str | None = None, cfg: dict | None = None):
    cfg = {**DEFAULTS, **(cfg or {})}
    path = xml or cfg["xml"]
    if not path.lstrip().startswith("<") and not Path(path).exists():
        near = Path(__file__).resolve().parent / path
        if near.exists():
            path = str(near)
    model = mujoco.MjModel.from_xml_path(path) if not path.lstrip().startswith("<") else mujoco.MjModel.from_xml_string(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, CrutchInterface(model, data, cfg)


# --------------------------------------------------------------------------
# 最小验收
# --------------------------------------------------------------------------
def _settle(ci, seconds, human_pos, human_vel=None, muscle=None):
    n = int(round(seconds / ci.control_dt))
    hv = np.zeros(3) if human_vel is None else np.asarray(human_vel, float)
    u = np.zeros(ci.muscle_ids.size) if muscle is None else muscle
    out = None
    for _ in range(n):
        out = ci.step(human_pos, hv, u)
    return out


def selftest() -> int:
    model, data, ci = load()
    d = ci.d
    print("[1] XML 独立编译与几何")
    v = ci.validate()
    for k in ("chassis_box_size", "chassis_mass_kg", "mass_ok", "rod_length_measured", "rod_length_ok",
              "handle_height_m", "handle_height_formula", "demo_hand_height_m", "demo_handle_minus_hand_m",
              "hand_height_ref_m", "rod_length_for_hand_ref_m", "handle_over_hand_ref_m",
              "crutch_mass_kg", "nq_nv_nu", "eq_inactive_at_start"):
        print(f"    {k:28s} {v[k]}")
    print(f"    -> {'PASS' if v['ok'] else 'FAIL'}")

    print("[2] 传感器名称与维度")
    print(f"    dims {v['sensor_dims']}  ok={v['sensor_dims_ok']}")
    for _ in range(5):
        mujoco.mj_step(model, d)
    print(f"    sample framepos={np.round(ci._sensor('chassis_pos', 3), 4)} framelinvel={np.round(ci._sensor('chassis_vel', 3), 4)}")
    print(f"    -> {'PASS' if v['sensor_dims_ok'] else 'FAIL'}")

    print("[3] 斜向速度指令限幅")
    model, data, ci = load(cfg={"forward_offset": 0.0, "lateral_offset": 0.0, "k_pos": 0.0, "k_vel": 0.0, "k_ff": 1.0})
    ci.mode, ci.affected_side, ci.active_side = "none", "left", None
    vh = 0.8 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    pos, prev_cmd = np.zeros(3), np.zeros(2)
    worst, n_lim, vs, dsens = 0.0, 0, [], []
    for _ in range(300):
        pos = pos + vh * ci.control_dt
        out = ci.step(pos, vh, np.zeros(0))
        worst = max(worst, float(np.linalg.norm(out["v_cmd"] - prev_cmd)))
        prev_cmd = out["v_cmd"].copy()
        n_lim += int(out["dv_limited"])
        vs.append(out["v_meas"].copy())
        dsens.append(ci._sensor("chassis_vel", 3)[:2].copy())
    limit = ci.cfg["cmd_dv_coef"] * ci.control_dt
    diagonal = abs(vs[-1][0] - vs[-1][1]) < 0.02 and vs[-1][0] > 0.5
    frame_delta = float(np.max(np.abs(np.asarray(dsens) - np.asarray(vs))))
    print(f"    control_dt={ci.control_dt:.4f}s  limit={limit:.5f}  worst|v_cmd-v_prev|={worst:.6f}  limited={n_lim}/300")
    print(f"    final v_cmd={np.round(prev_cmd, 4)} v_meas={np.round(vs[-1], 4)} diagonal={diagonal}")
    print(f"    framelinvel vs qvel max|diff|={frame_delta:.2e} (world-frame 一致性)")
    ok3 = worst <= limit * (1 + 1e-9) and diagonal and frame_delta < 1e-6
    print(f"    -> {'PASS' if ok3 else 'FAIL'}")

    print("[3b] 输出力限幅")
    model, data, ci = load()
    for _ in range(400):
        data.xfrc_applied[ci.body_chassis] = [-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data.ctrl[ci.act["crutch_x"]] = 2.0
        mujoco.mj_step(model, data)
    f_sat = float(data.actuator_force[ci.act["crutch_x"]])
    data.xfrc_applied[ci.body_chassis] = [0.0] * 6
    ok3b = abs(abs(f_sat) - 400.0) < 1e-6
    print(f"    对抗 1000 N、指令 v=2 m/s -> |actuator_force|={abs(f_sat):.2f} N（forcerange 400 N）")
    print(f"    -> {'PASS' if ok3b else 'FAIL'}")
    ok3 = ok3 and ok3b

    print("[4] 三种辅助模式的约束状态")
    model, data, ci = load()
    rows = []
    for aff in ("left", "right"):
        for mode in ("none", "ipsilateral", "contralateral"):
            r = ci.set_assistance(mode, aff, force=True)
            rows.append((aff, mode, r["active_side"], r["eq_active"]["left"], r["eq_active"]["right"]))
            print(f"    affected={aff:5s} {mode:14s} active={str(r['active_side']):5s} "
                  f"eq_l={r['eq_active']['left']} eq_r={r['eq_active']['right']}")
    ok4 = all((m == "none" and a is None) or (m != "none" and a is not None) for _, m, a, _, _ in rows) \
        and sum(1 for *_, l, rr in rows if l or rr) == 4

    print("[4b] 运行中错位时的重新激活保护")
    model, data, ci = load()
    ci.align_grip("left")
    ci.set_assistance("ipsilateral", "left", force=True)
    qadr = int(model.jnt_qposadr[ci.jnt_x])
    data.qpos[qadr + 1] += 0.30
    mujoco.mj_forward(model, data)
    r = ci.set_assistance("contralateral", "left")
    print(f"    chassis 偏移 0.30 m 后尝试激活健侧: ok={r['ok']} reason={r['reason']} gap={r['gap_m']:.3f} m "
          f"eq={r['eq_active']}")
    ok4b = (not r["ok"]) and not any(r["eq_active"].values())
    print(f"    -> {'PASS' if ok4b else 'FAIL'}")
    ok4 = ok4 and ok4b

    print("[5] 开启连接后的人机传力")
    model, data, ci = load(cfg={"forward_offset": 0.0, "lateral_offset": 0.0})
    ci.align_grip("left")
    p_hold = data.site_xpos[ci.site_hand["left"]][:2].copy()
    ci.tare_now()
    ci.set_assistance("ipsilateral", "left", force=True)
    out = _settle(ci, 4.0, np.array([*p_hold, 0.0]))
    m_stub = float(model.body_mass[ci.body_hand["left"]])
    expect = m_stub * float(np.abs(model.opt.gravity[2]))
    got = out["support_up_N"]
    print(f"    stub mass={m_stub:.3f} kg -> 期望 support={expect:.2f} N, 实测={got:.2f} N "
          f"(err {100 * (got - expect) / expect:+.2f}%)")
    print(f"    raw force(site frame)={np.round(out['grip_force_raw'], 2)}  world={np.round(out['grip_force_world'], 2)}")

    print("[5b] 已知附加载荷 / 偏航不变性 / 力矩通道")
    qy = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")])
    data.qpos[qy] = np.deg2rad(70.0)
    mujoco.mj_forward(model, data)
    yawed = _settle(ci, 3.0, np.array([*p_hold, 0.0]))["support_up_N"]
    stub = ci.body_hand["left"]
    data.xfrc_applied[stub] = [0.0, 0.0, -19.62, 0.0, 0.0, 0.0]
    extra = _settle(ci, 3.0, np.array([*p_hold, 0.0]))["support_up_N"]
    data.xfrc_applied[stub] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data.xfrc_applied[ci.body_handle] = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    tau = _settle(ci, 1.0, np.array([*p_hold, 0.0]))["grip_torque_raw"]
    data.xfrc_applied[ci.body_handle] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"    yaw 70deg 后 support={yawed:.2f} N（应不变）")
    print(f"    附加 -19.62 N 竖直载荷后 support={extra:.2f} N（应 +19.62）")
    print(f"    handle 施加 +5 N.m -> torque 读数 {np.round(tau, 3)}（球铰 grip 不传力矩，把手质心与 site 重合）")

    print("[5c] 关闭连接后残留")
    ci.set_assistance("none", "left")
    off = _settle(ci, 1.5, np.array([*p_hold, 0.0]))
    print(f"    support={off['support_up_N']:+.2f} N  eq={off['eq_active']}")

    ok5 = (abs(got - expect) / expect < 0.02 and got > 0
           and abs(yawed - expect) < 0.2
           and abs(extra - (expect + 19.62)) < 0.3
           and abs(tau[2] + 5.0) < 0.05
           and abs(off["support_up_N"]) < 0.5)
    print(f"    -> {'PASS' if ok5 else 'FAIL'}")

    passed = bool(v["ok"] and v["sensor_dims_ok"] and ok3 and ok4 and ok5)
    print("\n验收:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(selftest())
