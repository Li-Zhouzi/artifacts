# python3 multi_simulator.py --policy sia --interval 5 --oracle_num_nodes=16 --oracle_num_replicas=64 workloads/profile.csv > ./results/test_with_Xput.txt


python3 multi_simulator.py --policy sia --interval 5 --disable_bsz_tuning --oracle_num_nodes=16 --oracle_num_replicas=64 workloads/profile_cifar10.csv > ./results/test_fix_localbsz.txt



# python3 multi_simulator.py --policy dummy --b 9 --disable_bsz_tuning --oracle_num_nodes=16 --oracle_num_replicas=64 workloads/profile_cifar10.csv --interval 1 > ./results/test.txt