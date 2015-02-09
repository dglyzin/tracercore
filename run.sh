srun -N1 -n1 bin/HS test_N1_n1 res_N1_n1
srun -N2 -n2 bin/HS test_N2_n2 res_N2_n2

srun -N2 -n2 bin/HS test_N2_n2_CPU1_GPU3 res_N2_n2_CPU1_GPU3