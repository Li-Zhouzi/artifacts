#! /bin/bash
#
#
echo "name,epoch,replicas,goodput,throughput,efficiency" > ${2}/${1}gpu.csv
# python3 multi_simulator.py --policy gavel --oracle_num_nodes=16 --oracle_num_replicas=64 workloads/test.csv | grep -e "^speedup " | sed 's/speedup //' | sed 's/ /,/g' >> ${1}.csv
python3 multi_simulator.py --policy dummy --b ${1} --oracle_num_nodes=16 --oracle_num_replicas=64 workloads/profile.csv --interval 5 | grep -e "^speedup " | sed 's/speedup //' | sed 's/ /,/g' >> ${2}/${1}gpu.csv