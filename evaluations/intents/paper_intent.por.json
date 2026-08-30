{
  "_comment": "PAPER OF RECORD intent (Tom 2026-08-30). Real deployments; everything should cache-hit since the campaigns are done. n_prefixes is a SHIM for now: prefixes 2,3,5 on the actual-3 deployment at 10 iters. All artifacts land locally in figures/paper_artifacts.",
  "run_id": "paperv1",
  "where": "vm",
  "storage_vm": "i-0428c395787bc3ca0",
  "iters": 150,
  "env": {
    "SCULPTOR_DEPSTORE": "1",
    "SCULPTOR_LOG_MEM": "0"
  },
  "stages": {
    "deployment_sizes": {
      "enabled": true,
      "dpsizes": [
        5,
        10,
        15,
        20,
        25,
        "actual-32"
      ],
      "nsim": [
        20,
        20,
        12,
        5,
        4,
        3
      ],
      "cache_fn": "cache/cluster_runs/20260822_220131-prefixbudget3/metrics_by_dpsize.pkl",
      "figures_subdir": "cluster/20260822_220131-prefixbudget3",
      "plot": true,
      "port": 31700,
      "env": {},
      "artifacts": {
        "src_dir": "/home/ubuntu/sparse_advertisements_code/paper_artifacts/eods",
        "files": [
          "average_latency_over_deployment_size_normal.pdf",
          "average_latency_over_deployment_size_fail_ingress_mlu.pdf",
          "average_latency_over_deployment_size_fail_site_mlu.pdf",
          "average_congestion_over_deployment_size_fail_ingress_mlu.pdf",
          "average_congestion_over_deployment_size_fail_site_mlu.pdf",
          "average_high_latency_over_deployment_size_fail_ingress_mlu.pdf",
          "average_high_latency_over_deployment_size_fail_site_mlu.pdf",
          "flash_crowd_blowup_before_congestion_over_deployment_size.pdf",
          "diurnal_blowup_before_congestion_over_deployment_size.pdf",
          "percent_traffic_within_10_ms_site_failure_over_deployment_size.pdf",
          "percent_traffic_within_50_ms_normal_over_deployment_size.pdf"
        ],
        "dst": "figures/paper_artifacts"
      },
      "outputs": [
        "figures/cluster/20260822_220131-prefixbudget3/*over_deployment_size*.pdf"
      ]
    },
    "n_prefixes": {
      "enabled": true,
      "_comment": "SHIM scale for now (Tom 2026-08-30)",
      "dpsize": "testing_feature-actual-3",
      "prefixes": [
        2,
        3,
        5
      ],
      "nsim": 1,
      "iters": 10,
      "cache_fn": "cache/paperv1/prefix_metrics.pkl",
      "figures_subdir": "cluster/prefix_sweep",
      "plot": true,
      "port": 31702,
      "env": {},
      "artifacts": {
        "src_dir": "/home/ubuntu/sparse_advertisements_code/paper_artifacts/n_prefixes",
        "files": [],
        "dst": "figures/paper_artifacts"
      },
      "outputs": [
        "figures/cluster/prefix_sweep/*over_prefix_budget*.pdf"
      ]
    },
    "paper_table": {
      "enabled": true,
      "dpsize": 32,
      "nsim": 1,
      "run_tag": "20260823_130342_papertable32b",
      "objectives": [
        "avg_latency",
        "joint_priority",
        "frac_beyond_optimal",
        "max_util",
        "per_site_cost"
      ],
      "out": "figures/cluster/20260823_130342-papertable32b/paper_table_full",
      "port": 31704,
      "env": {},
      "artifacts": {
        "src_dir": "/home/ubuntu/sparse_advertisements_code/paper_artifacts/paper_table",
        "files": [
          "paper_table.csv",
          "paper_table.tex",
          "paper_table_key.csv",
          "paper_table_key.tex"
        ],
        "dst": "figures/paper_artifacts"
      },
      "outputs": [
        "figures/cluster/20260823_130342-papertable32b/paper_table_full/*"
      ]
    },
    "hardness_figures": {
      "enabled": false,
      "kind": "local_artifact",
      "artifacts": {
        "local_dir": "figures/dashboards/ablation_scout",
        "files": [
          "grid_objdim_5panel.png",
          "ablation_scout_grid_bars.png",
          "ablation_scout_difficulty_scatter.png"
        ],
        "dst": "figures/paper_artifacts"
      },
      "runbook": "SCULPTOR_LOG_MEM=0 python run_ablation_grid.py --number_measurements_allowed '[5,10]' --deployments 3 --num_iters 250 --objectives all --dpsize small && python -m dashboard.plot_ablation_scout"
    }
  }
}