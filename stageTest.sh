srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 3 10 save/state1 nothing
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state2 save/state1
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state3 save/state2
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state4 save/state3
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state5 save/state4
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state6 save/state5
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state7 save/state6
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state8 save/state7
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 7 10 save/state9 save/state8
srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3 5 10 nothing save/state9