python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v4/no_noise_rls_only --ntrials 100 --noise none --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v4/low_noise_rls_only --ntrials 100 --noise low --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v4/medium_noise_rls_only --ntrials 100 --noise medium --save_traj

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v4/high_noise_rls_only --ntrials 100 --noise high --save_traj

