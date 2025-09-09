python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v2/no_noise --ntrials 100 --noise none --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v2/low_noise --ntrials 100 --noise low --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v2/medium_noise --ntrials 100 --noise medium --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v2/high_noise --ntrials 100 --noise high --save_traj

