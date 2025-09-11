python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp3_v0/medium_noise_sub1 --ntrials 100 --noise medium --save_traj \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 1

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp3_v0/medium_noise_sub2 --ntrials 100 --noise medium --save_traj \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 2

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp3_v0/medium_noise_sub3 --ntrials 100 --noise medium --save_traj \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 3

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp3_v0/medium_noise_sub4 --ntrials 100 --noise medium --save_traj \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 4
