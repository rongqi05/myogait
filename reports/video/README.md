# 视频清单（`reports/video/`）

参考：`CMU/120/cmu120_19_straight_h000`（未拼接、正向、heading 0°、从原点起步、
直线窗 10.41 s、速度 0.457 m/s）
checkpoint：`amathislab/mm-10m-2` update 12500 ｜ 模式 stochastic ｜
起点 `--start_step 53`（对齐直线窗起点）｜ 20 fps（`--video_every 5`）

| 文件 | 时长 | 结果 | 侧偏 | 速度 | 骨盆倾角 | 骨盆最低 |
|---|---:|---|---:|---:|---:|---:|
| `cmu120_19_best_10s_seed17_lat0.112.mp4` | **10.00 s** | **time_limit_success** | **0.112 m** | 0.454 m/s | 11.4° | 0.902 m |

（另有较早的 10 s 验收阶段视频与其它回放视频，见本目录其余 `.mp4`。）

## 怎么读这段

- **best 10 s（seed 17）**：走满 10.00 s，侧偏 0.112 m，**低于 0.35 m 阈值**。
  这是目前单个回合里表现最好的一次。它**只**不满足冻结的 0.6 m/s 速度门槛
  （参考本身只有 0.457 m/s，代数上不可能达到 0.6）。
  换言之：在这条参考上，"10 s 连续行走 + 侧偏合格"是能做到的。

## 重要限制（不要用这些视频宣称验收通过）

1. 这是**挑选出来的单回合**，不是 20 种子统计。统计结论见 git 历史中的报告 N：
   6 s → 20/20；8 s → 20/20；10 s → 18/20（2 次跌倒）。
2. 该参考的直线窗只有 **10.41 s**，所以 10 s 评估已接近它的上限，**不能**据此做 20 s 评估。
3. 参考速度 0.457 m/s，低于冻结的 0.6 m/s 门槛，因此 **C（原定直线验收）必然为 0**。
4. 视频只用于人工核对姿态与接触；足滑/接触等量化证据在
   `reports/qc_CMU_120_cmu120_19_straight_h000.json`。

## 复现命令

```bash
export PYTHONPATH="external/MuscleMimic:scripts"
CKPT=artifacts/checkpoints/checkpoint_12500

.venv/bin/python scripts/eval_walk.py --checkpoint "$CKPT" \
  --motion_path CMU/120/cmu120_19_straight_h000 --mode stochastic \
  --seeds 17 --seconds 10 --start_step 53 --video --video_every 5 \
  --out_dir runs/VID_best10s
```
