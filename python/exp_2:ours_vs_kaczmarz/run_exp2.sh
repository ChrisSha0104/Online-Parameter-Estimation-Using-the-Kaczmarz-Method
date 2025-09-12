python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
    --outdir paper_plots/exp2_v1/no_noise_wind5 --ntrials 32 --noise none --save_traj --window 5

python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
    --outdir paper_plots/exp2_v1/low_noise_wind5 --ntrials 32 --noise low --save_traj --window 5

python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
    --outdir paper_plots/exp2_v1/medium_noise_wind5 --ntrials 32 --noise medium --save_traj --window 5

python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
    --outdir paper_plots/exp2_v1/high_noise_wind5 --ntrials 32 --noise high --save_traj --window 5


# python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
#     --outdir paper_plots/exp2_v0/medium_noise_freq10 --ntrials 100 --noise medium --save_traj --est_freq 10

# python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
#     --outdir paper_plots/exp2_v0/medium_noise_freq5 --ntrials 100 --noise medium --save_traj --est_freq 5

# python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
#     --outdir paper_plots/exp2_v0/medium_noise_freq50 --ntrials 100 --noise medium --save_traj --est_freq 50

# python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
#     --outdir paper_plots/exp2_v0/medium_noise_freq2 --ntrials 100 --noise medium --save_traj --est_freq 2

# python python/exp_2:ours_vs_kaczmarz/run_exp2_parallel.py \
#     --outdir paper_plots/exp2_v0/medium_noise_freq1 --ntrials 100 --noise medium --save_traj --est_freq 1