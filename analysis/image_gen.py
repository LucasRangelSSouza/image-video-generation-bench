"""SDXL-Turbo (512x512, 4 steps) and FLUX.3 Klein (1024x1024, 16 steps): seconds and dollars per image, batching gain, cold start."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchkit import usd_per_thousand
from common import BLUE, GREEN, ORANGE, RED, gpu_name, read, save_fig, save_table, update_summary

libs = read("libs_runs.csv")
libs["gpu"] = libs.gpu.map(gpu_name)
s = libs[(libs.lib == "sdxl_turbo") & (libs.res == "512x512")]
t = s.pivot_table(index="gpu", columns="batch", values="s_per_image", aggfunc="median")
t.columns = ["batch1_s_per_image", "batch4_s_per_image"]
t["batching_gain"] = t.batch1_s_per_image / t.batch4_s_per_image
dph = s.groupby("gpu").dph_usd.median()
t["usd_per_1000_images_batch4"] = [usd_per_thousand(dph[g], t.loc[g, "batch4_s_per_image"]) for g in t.index]
t["first_call_s"] = s.groupby("gpu").first_call_s.median()
t = t.sort_values("batch4_s_per_image")
save_table(t.round(3), "sdxl_turbo_by_gpu")

fl = read("media_gated_runs.csv")
fl = fl[fl.tool.str.startswith("FLUX") & (fl.metric == "wall_s")].copy()
fl["gpu"] = fl.gpu.map(gpu_name)
flux = fl.groupby("gpu").value.median().sort_values()
failed = read("media_gated_runs.csv")
failed = failed[failed.tool.str.startswith("FLUX") & failed.metric.str.startswith("failed")]
save_table(flux.rename("median_wall_s_per_image").round(1).to_frame(), "flux3_klein_by_gpu")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
x = np.arange(len(t))
axes[0].bar(x - 0.2, t.batch1_s_per_image, 0.4, color=ORANGE, label="batch 1")
axes[0].bar(x + 0.2, t.batch4_s_per_image, 0.4, color=BLUE, label="batch 4")
axes[0].set_xticks(x)
axes[0].set_xticklabels(t.index, rotation=35, fontsize=8)
axes[0].set_ylabel("Seconds per image, SDXL-Turbo 512x512")
axes[0].legend(frameon=False)
axes[1].bar(t.index, t.usd_per_1000_images_batch4, color=GREEN)
axes[1].set_ylabel("US$ per 1,000 images (batch 4)")
axes[1].tick_params(axis="x", rotation=35, labelsize=8)
axes[2].bar(flux.index, flux.values, color=BLUE)
axes[2].set_ylabel("Seconds per 1024x1024 image, FLUX.3 Klein")
axes[2].tick_params(axis="x", rotation=35, labelsize=8)
for g in failed.gpu.map(gpu_name):
    axes[2].text(0, flux.max() * 0.9, f"{g}: FP8 matmul needs compute capability 8.9, run failed", fontsize=7, color=RED)
save_fig(fig, "image_generation.png")

update_summary({"sdxl_fastest_gpu": t.index[0], "sdxl_fastest_s": t.batch4_s_per_image.iloc[0], "sdxl_cheapest_gpu": t.usd_per_1000_images_batch4.idxmin(), "sdxl_cheapest_usd_per_1000": t.usd_per_1000_images_batch4.min(),
                "sdxl_gain_min": t.batching_gain.min(), "sdxl_gain_max": t.batching_gain.max(), "sdxl_gain_gb10": t.loc["GB10", "batching_gain"],
                "flux_best_s": flux.iloc[0], "flux_best_gpu": flux.index[0], "flux_gb10_s": flux["GB10"], "flux_failed_gpu": gpu_name(failed.gpu.iloc[0])})
print(t.round(3)); print(flux.round(1))
