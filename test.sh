#!/bin/bash

# DMC domain run example for SAR with DrQ + SAC backend
# Use this script to launch SAR_test.py with appropriate arguments

python launch.py \
--env dmc.cheetah.run \
--agent drq \
--base sac \
--auxiliary sar \
--num_sources 2 \
--dynamic -bg -tbg \
--disenable_default \
--critic_lr 5e-4 \
--actor_lr 5e-4 \
--alpha_lr 5e-4 \
--extr_lr 5e-4 \
--targ_extr 0 \
--nstep_of_rsd 5 \
--num_sample 256 \
--batch_size 256 \
--opt_mode max \
--omega_opt_mode min_mu \
--rs_fc \
--discount_of_rs 0.8 \
--extr_update_via_qfloss True \
--cuda_id 0 \
--exp_name test_sar_run \
-s 0
