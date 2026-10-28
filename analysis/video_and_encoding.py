"""LTX-Video generation time per second of video, model load time, and FFmpeg encoding: CPU x264 against the GPU NVENC unit."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchkit import usd_per_thousand
from common import BLUE, GREEN, ORANGE, PURPLE, gpu_name, read, save_fig, save_table, update_summary

libs = read("libs_runs.csv")
libs["gpu"] = libs.gpu.map(gpu_name)

lx = libs[(libs.lib == "ltx_video") & libs.gen_s_per_video_s.notna()]
warm = lx[lx.frames == 49]  # the 25-frame clip is the first call and includes warm-up
lt = warm.groupby("gpu").agg(gen_s_per_video_s=("gen_s_per_video_s", "median"), load_s=("load_s", "median"), dph=("dph_usd", "median")).sort_values("gen_s_per_video_s")
lt["usd_per_video_minute"] = lt.gen_s_per_video_s * 60 * lt.dph / 3600
lt["usd_to_load_model"] = lt.load_s * lt.dph / 3600
save_table(lt.round(3), "ltx_video_by_gpu")

ff = libs[(libs.lib == "ffmpeg") & (libs.status == "ok")]
enc = ff.pivot_table(index="gpu", columns=["res", "codec"], values="speed_x", aggfunc="median")
save_table(enc.round(2), "ffmpeg_speed_x_realtime")
h = enc["1080p"]
gain = (h["h264_nvenc"] / h["libx264"]).dropna()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(lt.index, lt.gen_s_per_video_s, color=BLUE)
axes[0].set_ylabel("Seconds to generate 1 s of video (LTX-Video 512x320)")
axes[1].bar(lt.index, lt.load_s, color=ORANGE)
axes[1].set_ylabel("Model load time (s)")
for ax in axes[:2]:
    ax.tick_params(axis="x", rotation=40, labelsize=8)
xx = np.arange(len(h))
axes[2].bar(xx - 0.27, h["libx264"], 0.27, color=GREEN, label="x264 (CPU)")
axes[2].bar(xx, h["h264_nvenc"], 0.27, color=BLUE, label="H.264 NVENC")
axes[2].bar(xx + 0.27, h["hevc_nvenc"], 0.27, color=PURPLE, label="HEVC NVENC")
axes[2].set_xticks(xx)
axes[2].set_xticklabels(h.index, rotation=40, fontsize=8)
axes[2].set_ylabel("1080p encoding speed (x real time)")
axes[2].legend(frameon=False, fontsize=7)
save_fig(fig, "video_and_encoding.png")

update_summary({"ltx_best_gpu": lt.index[0], "ltx_best_s": lt.gen_s_per_video_s.iloc[0], "ltx_gb10_s": lt.loc["GB10", "gen_s_per_video_s"], "ltx_load_min_s": lt.load_s.min(), "ltx_load_max_s": lt.load_s.max(),
                "ltx_load_over_gen": (lt.load_s / lt.gen_s_per_video_s).median(), "nvenc_over_x264_min": gain.min(), "nvenc_over_x264_max": gain.max(),
                "x264_4090_1080p": h.loc["RTX 4090", "libx264"], "x264_5090_1080p": h.loc["RTX 5090", "libx264"]})
print(lt.round(2)); print(gain.round(2))
