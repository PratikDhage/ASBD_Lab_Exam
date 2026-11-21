# Figure Placement Checklist

Copy each source file to the `report/images/` directory with the target filename.

| # | LaTeX Figure Label | Target Filename (in `images/`) | Source File Path |
|---|---|---|---|
| 1 | `fig:hist_kde` | `01_histogram_stars_kde.png` | `outputs/visualizations/01_histogram_stars_kde.png` |
| 2 | `fig:boxplot` | `02_boxplot_review_length_by_rating.png` | `outputs/visualizations/02_boxplot_review_length_by_rating.png` |
| 3 | `fig:violin` | `03_violinplot_reputation_by_rating.png` | `outputs/visualizations/03_violinplot_reputation_by_rating.png` |
| 4 | `fig:barchart` | `04_barchart_top_recipes.png` | `outputs/visualizations/04_barchart_top_recipes.png` |
| 5 | `fig:pareto` | `05_pareto_chart_recipes.png` | `outputs/visualizations/05_pareto_chart_recipes.png` |
| 6 | `fig:linechart` | `06_linechart_temporal_trends.png` | `outputs/visualizations/06_linechart_temporal_trends.png` |
| 7 | `fig:scatter` | `07_scatterplot_thumbs_vs_bestscore.png` | `outputs/visualizations/07_scatterplot_thumbs_vs_bestscore.png` |
| 8 | `fig:bubble` | `08_bubblechart_recipe_engagement.png` | `outputs/visualizations/08_bubblechart_recipe_engagement.png` |
| 9 | `fig:heatmap` | `09_heatmap_correlation_matrix.png` | `outputs/visualizations/09_heatmap_correlation_matrix.png` |
| 10 | `fig:hexbin` | `10_hexbin_thumbs_density.png` | `outputs/visualizations/10_hexbin_thumbs_density.png` |
| 11 | `fig:wordcloud` | `11_wordcloud_review_text.png` | `outputs/visualizations/11_wordcloud_review_text.png` |
| 12 | `fig:stacked` | `12_stacked_bar_rating_by_engagement.png` | `outputs/visualizations/12_stacked_bar_rating_by_engagement.png` |
| 13 | `fig:donut` | `13_donut_chart_reputation_segments.png` | `outputs/visualizations/13_donut_chart_reputation_segments.png` |
| 14 | `fig:network` | `14_network_graph_association_rules.png` | `outputs/visualizations/14_network_graph_association_rules.png` |
| 15 | `fig:arm_scatter` | `arm_support_vs_confidence.png` | `outputs/association_rules/arm_support_vs_confidence.png` |
| 16 | `fig:arm_lift` | `arm_top_rules_lift_barchart.png` | `outputs/association_rules/arm_top_rules_lift_barchart.png` |
| 17 | `fig:knn_perf` | `knn_performance_vs_k.png` | `outputs/knn/knn_performance_vs_k.png` |
| 18 | `fig:knn_cm` | `knn_confusion_matrix.png` | `outputs/knn/knn_confusion_matrix.png` |
| 19 | `fig:dendro` | `agnes_dendrogram_ward.png` | `outputs/clustering/agnes_dendrogram_ward.png` |
| 20 | `fig:pca_scatter` | `agnes_pca_cluster_scatter.png` | `outputs/clustering/agnes_pca_cluster_scatter.png` |
| 21 | `fig:fi_cfi` | `aclose_fi_vs_cfi_comparison.png` | `outputs/closed_itemsets/aclose_fi_vs_cfi_comparison.png` |
| 22 | `fig:redundancy` | `aclose_redundancy_reduction_curve.png` | `outputs/closed_itemsets/aclose_redundancy_reduction_curve.png` |
| 23 | `fig:bench_time` | `benchmark_execution_times.png` | `outputs/benchmarks/benchmark_execution_times.png` |
| 24 | `fig:bench_speed` | `benchmark_speedup_ratios.png` | `outputs/benchmarks/benchmark_speedup_ratios.png` |

## Quick Copy Command (PowerShell)

Run from the `d:\softwares\ASBD` directory:

```powershell
# Copy all 14 descriptive visualizations
Copy-Item outputs\visualizations\*.png report\images\

# Copy algorithm-specific plots
Copy-Item outputs\association_rules\*.png report\images\
Copy-Item outputs\knn\*.png report\images\
Copy-Item outputs\clustering\*.png report\images\
Copy-Item outputs\closed_itemsets\*.png report\images\
Copy-Item outputs\benchmarks\*.png report\images\
```

After running the above, all 24 images will be in `report\images\` and the LaTeX report will compile with all figures displayed.
