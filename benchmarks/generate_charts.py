# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np

# 1. Real Data collected from the benchmark runs
# We use the average of the 6 benchmark categories to provide a representative comparison
models = ['Gemma4:e2b', 'Gemma4:e4b', 'Gemma4:12b']

# Average Generation Throughput (Tokens/second)
tpm_tps = [119.54, 68.97, 31.12]
ollama_tps = [96.67, 57.73, 26.72]

# Average Time-to-First-Token (seconds)
tpm_ttft = [0.22, 0.36, 0.90] # 221ms, 358ms, 896ms in seconds
ollama_ttft = [7.96, 9.64, 32.73] # 7956ms, 9635ms, 32733ms in seconds

# 2. Setup styles
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(models))
width = 0.35

# Color Palettes
tpm_color = '#00d2c4'   # Neon Teal
ollama_color = '#9c27b0' # Purple / Indigo

# --- Plot 1: Throughput ---
rects1_1 = ax1.bar(x - width/2, tpm_tps, width, label='TPM-MLX (Ours)', color=tpm_color, edgecolor='none', alpha=0.9)
rects1_2 = ax1.bar(x + width/2, ollama_tps, width, label='Ollama', color=ollama_color, edgecolor='none', alpha=0.9)

ax1.set_ylabel('Throughput (Tokens/Second)', fontsize=12, fontweight='bold', color='#e0e0e0')
ax1.set_title('Generation Throughput (Higher is Better)', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, color='#e0e0e0')
ax1.legend(frameon=True, facecolor='#121212', edgecolor='#2c2c2c', fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#444444')

# Add values above bars
def autolabel_tps(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f} t/s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#ffffff', fontweight='semibold')

autolabel_tps(rects1_1, ax1)
autolabel_tps(rects1_2, ax1)

# --- Plot 2: Latency ---
rects2_1 = ax2.bar(x - width/2, tpm_ttft, width, label='TPM-MLX (Ours)', color=tpm_color, edgecolor='none', alpha=0.9)
rects2_2 = ax2.bar(x + width/2, ollama_ttft, width, label='Ollama', color=ollama_color, edgecolor='none', alpha=0.9)

ax2.set_ylabel('Time-To-First-Token (Seconds)', fontsize=12, fontweight='bold', color='#e0e0e0')
ax2.set_title('Time-To-First-Token Latency (Lower is Better)', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11, color='#e0e0e0')
ax2.legend(frameon=True, facecolor='#121212', edgecolor='#2c2c2c', fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.3, color='#444444')

# Add values above bars
def autolabel_ttft(rects, ax):
    for rect in rects:
        height = rect.get_height()
        # If less than 1 second, display in ms
        label_text = f'{height*1000:.0f}ms' if height < 1.0 else f'{height:.2f}s'
        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#ffffff', fontweight='semibold')

autolabel_ttft(rects2_1, ax2)
autolabel_ttft(rects2_2, ax2)

# Global styling enhancements
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    ax.tick_params(colors='#e0e0e0')

plt.suptitle('TPM-MLX vs Ollama Performance Benchmarks (Apple M4 Pro)', fontsize=16, fontweight='bold', color='#ffffff', y=0.98)
plt.tight_layout()

# Save the plot
output_path = 'tpm_mlx_vs_ollama_benchmarks.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#121212')
print(f"Chart saved to {output_path}")
