python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v3/no_noise --ntrials 100 --noise none --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v3/low_noise --ntrials 100 --noise low --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v3/medium_noise --ntrials 100 --noise medium --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v3/high_noise --ntrials 100 --noise high --save_traj

