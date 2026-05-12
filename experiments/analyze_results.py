import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from matplotlib.ticker import AutoMinorLocator

def analyze_and_plot():
    # 데이터셋 순서 고정
    target_datasets = ['ml-20m', 'yelp2018',"gowalla"]
    case_types = [
        ('best', 'Best Optimization (by valid_ndcg_100)'),
        ('lambda0.01', 'Fixed $\lambda=0.01$')
    ]
    
    for case_id, case_title in case_types:
        print(f"\nGenerating Integrated Plot with Insets for Case: {case_id}...")
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        
        for i, ds in enumerate(target_datasets):
            ax = axes[i]
            pattern = f'exp_result/grid_search_ASPIRE_EASE_{ds}_strong.csv'
            files = glob.glob(pattern)
            
            if not files:
                ax.text(0.5, 0.5, f'Data not found:\n{ds}', ha='center', va='center')
                continue
                
            df = pd.read_csv(files[0])
            
            if case_id == 'best':
                # 최적화 기준 지표를 valid_ndcg_100으로 변경
                val_metric = 'valid_ndcg_100'
                idx = df.groupby('alpha')[val_metric].idxmax()
                plot_df = df.loc[idx].sort_values('alpha')
            else:
                plot_df = df[np.isclose(df['reg_lambda'], 0.01, atol=1e-5)].sort_values('alpha')

            if plot_df.empty:
                ax.text(0.5, 0.5, f'No data for {case_id}\nin {ds}', ha='center', va='center')
                continue

            # Save plot data to CSV
            csv_output_path = f"exp_result/summary_ASPIRE_EASE_{ds}_strong_{case_id}_at100.csv"
            plot_df.to_csv(csv_output_path, index=False)
            print(f"Saved plot data to {csv_output_path}")

            m = 'Recall@20'
            if m in plot_df.columns:
                alphas = plot_df['alpha'].values
                metrics_val = plot_df[m].values
                y_baseline = plot_df[plot_df['alpha'] == 0][m].iloc[0]
                y_max = metrics_val.max()

                # --- 1. 메인 그래프 그리기 (정직한 Linear Scale) ---
                ax.plot(alphas, metrics_val, label='ASPIRE-EASE', marker='o', color='blue', linewidth=2, markersize=4)
                ax.axhline(y_baseline, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
                
                ax.set_ylim(0, y_max * 1.1)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # --- 2. 확대도 (Inset Plot) 추가 (왼쪽 하단으로 이동) ---
                # 위치 설정: [left, bottom, width, height]
                ax_inset = ax.inset_axes([0.1, 0.1, 0.45, 0.45])
                
                # 확대도에도 데이터 플롯
                ax_inset.plot(alphas, metrics_val, marker='o', color='blue', linewidth=2, markersize=5)
                ax_inset.axhline(y_baseline, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
                
                # 핵심 구간(U-커브)으로 줌 설정
                ax_inset.set_xlim(-0.05, 1.05)
                y_inset_min = min(y_baseline, metrics_val[alphas <= 1.0].min())
                y_inset_max = max(y_baseline, metrics_val[alphas <= 1.0].max())
                margin = (y_inset_max - y_inset_min) * 0.2
                ax_inset.set_ylim(y_inset_min - margin, y_inset_max + margin)
                
                ax_inset.grid(True, linestyle='--', alpha=0.4)
                ax_inset.set_title("Zoomed U-Curve", fontsize=10, pad=5)
                
                # 메인 축에서 확대된 영역을 표시하는 연결선
                ax.indicate_inset_zoom(ax_inset, edgecolor="gray", alpha=0.3)

            ax.set_title(f"{ds.upper()}", fontsize=16, fontweight='bold')
            ax.set_xlabel('Alpha (Gamma)', fontsize=12)
            ax.set_ylabel('Recall@20', fontsize=12)
            # 범례 위치를 오른쪽 상단으로 변경
            if i == 0: ax.legend(loc='upper right')
            
        plt.suptitle(f"Performance Profile (Recall@20) - {case_title}", fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = f"exp_result/integrated_plot_{case_id}_recall_at20_tuned_at100.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"Academic format plot saved to: {output_path}")

if __name__ == "__main__":
    analyze_and_plot()
