#!/usr/bin/env bash
set -uo pipefail
cd /home/yiranwang/combinartorics/TriSearch-Calabi-Yau
export CY_EVAL_CPU_NO_CUDA_PROBE=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/yiranwang/combinartorics/TriSearch-Calabi-Yau
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=3 /home/yiranwang/anaconda3/envs/sage/bin/python -u scripts/eval_cy.py --config /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/configs/entropy_coef_0_01_rl_value_beam_search_gpu.json --setup_path /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/data/setups/h11_50_5_polytopes_1_start_seed_0 --output_dir /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/results/entropy_coef_0_01_rl_value_beam_search_20261005_184214 --cpu_ids 8 9 10 11 12 13 14 15 52 53 54 55 56 57 58 59 --plot_results > /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/entropy_coef_0_01_rl_value_beam_search_20261005_184214.log 2>&1 &
job_0_pid=$!
printf "%s %s\n" entropy_coef_0_01_rl_value_beam_search "$job_0_pid" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/pids_20261005_184214.txt
CUDA_VISIBLE_DEVICES=4 /home/yiranwang/anaconda3/envs/sage/bin/python -u scripts/eval_cy.py --config /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/configs/entropy_coef_0_01_rl_value_best_first_gpu.json --setup_path /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/data/setups/h11_50_5_polytopes_1_start_seed_0 --output_dir /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/results/entropy_coef_0_01_rl_value_best_first_20261005_184214 --cpu_ids 16 17 18 19 20 21 22 23 60 61 62 63 64 65 66 67 --plot_results > /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/entropy_coef_0_01_rl_value_best_first_20261005_184214.log 2>&1 &
job_1_pid=$!
printf "%s %s\n" entropy_coef_0_01_rl_value_best_first "$job_1_pid" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/pids_20261005_184214.txt
CUDA_VISIBLE_DEVICES=5 /home/yiranwang/anaconda3/envs/sage/bin/python -u scripts/eval_cy.py --config /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/configs/num_states_512_rl_value_beam_search_gpu.json --setup_path /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/data/setups/h11_50_5_polytopes_1_start_seed_0 --output_dir /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/results/num_states_512_rl_value_beam_search_20261005_184214 --cpu_ids 24 25 26 27 28 29 30 31 68 69 70 71 72 73 74 75 --plot_results > /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/num_states_512_rl_value_beam_search_20261005_184214.log 2>&1 &
job_2_pid=$!
printf "%s %s\n" num_states_512_rl_value_beam_search "$job_2_pid" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/pids_20261005_184214.txt
CUDA_VISIBLE_DEVICES=6 /home/yiranwang/anaconda3/envs/sage/bin/python -u scripts/eval_cy.py --config /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/configs/num_states_512_rl_value_best_first_gpu.json --setup_path /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/data/setups/h11_50_5_polytopes_1_start_seed_0 --output_dir /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/results/num_states_512_rl_value_best_first_20261005_184214 --cpu_ids 32 33 34 35 36 37 38 39 76 77 78 79 80 81 82 83 --plot_results > /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/num_states_512_rl_value_best_first_20261005_184214.log 2>&1 &
job_3_pid=$!
printf "%s %s\n" num_states_512_rl_value_best_first "$job_3_pid" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/pids_20261005_184214.txt
overall_code=0
wait "$job_0_pid"
job_0_code=$?
printf "%s %s\n" entropy_coef_0_01_rl_value_beam_search "$job_0_code" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/exit_codes_20261005_184214.txt
if (( job_0_code != 0 )); then overall_code=1; fi
wait "$job_1_pid"
job_1_code=$?
printf "%s %s\n" entropy_coef_0_01_rl_value_best_first "$job_1_code" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/exit_codes_20261005_184214.txt
if (( job_1_code != 0 )); then overall_code=1; fi
wait "$job_2_pid"
job_2_code=$?
printf "%s %s\n" num_states_512_rl_value_beam_search "$job_2_code" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/exit_codes_20261005_184214.txt
if (( job_2_code != 0 )); then overall_code=1; fi
wait "$job_3_pid"
job_3_code=$?
printf "%s %s\n" num_states_512_rl_value_best_first "$job_3_code" >> /home/yiranwang/combinartorics/TriSearch-Calabi-Yau/eval/sweep/checkpoint_600_h11_50/logs/exit_codes_20261005_184214.txt
if (( job_3_code != 0 )); then overall_code=1; fi
exit "$overall_code"
