import collections
import logging
import math
import numpy as np

from utils import LogWrapper


class DummyPolicy(object):
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        
        self.logger = LogWrapper(logging.getLogger(__name__))
        # self.logger.debug(num_gpus)
        # raise Exception()
        # width[application][epoch]=k
    
    def populate_valid_configs(*args):
        pass
    
    def optimize(self, jobs, nodes, prev_allocations):
        num_replicas = {}
        for key, job in jobs.items():
            num_replicas[key] = (
                self.num_gpus # math.ceil(job.target_batch_size / job.application.max_local_bsz)
            )

        job_allocs = dict()
        cluster_allocs = dict()
        for key, job in jobs.items():
            job_allocs[key] = ("aws", num_replicas[key])
            if "aws" not in cluster_allocs:
                cluster_allocs["aws"] = dict()
            cluster_allocs["aws"][key] = num_replicas[key]

        # cluster-specific job placements
        cluster_job_placements = dict()
        for cluster, cluster_nodes in nodes.items():
            node_remaining_gpus = np.asarray(
                [
                    node.resources["nvidia.com/gpu"]
                    for idx, node in nodes[cluster].items()
                ],
                dtype=np.uint32,
            )
            self.logger.debug("node remaining gpus: ", node_remaining_gpus)
            if cluster in cluster_allocs:
                cluster_job_placements[cluster] = self.alloc_to_placement(
                    cluster_allocs[cluster], node_remaining_gpus
                )
            else:
                cluster_job_placements[cluster] = dict()

        # merge allocs
        job_placements = {}
        for k, v in job_allocs.items():
            if v is None:
                job_placements[k] = (None, ())
            else:
                cluster_name, alloc = v
                job_placements[k] = (
                    cluster_name,
                    cluster_job_placements[cluster_name][k],
                )

        return job_placements

    def predict_step_time(self, job, num_replicas):
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1
        if num_replicas == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
        # throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
        # return atomic_bsz * count / throughput
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)
        return step_time + (step_time - sync_time) * accum_steps

    def alloc_to_placement(self, job_allocs, node_remaining_gpus):
        # job_allocs[job_name] = num_gpus
        # print(node_remaining_gpus)
        max_num_nodes = len(node_remaining_gpus)
        cur_node_id = 0
        placements = {}
        # sort by num gpus needed
        job_order = sorted(list(job_allocs.items()), key=lambda x: x[1], reverse=True)
        ngpus_per_node = np.max(node_remaining_gpus)
        # print("job order: ",job_order)
        # try to alloc distributed jobs on different nodes first
        for jobname, alloc in job_order:
            self.logger.debug(f"Allocating for job {jobname}, gpus needed {alloc}")
            num_gpus = alloc
            node_id = cur_node_id
            num_full_nodes = num_gpus // ngpus_per_node

            # corner case
            job_placement = []
            if num_gpus == 0:
                self.logger.warning(
                    f"Job {jobname} is not allocated any gpus because there are none left"
                )
                placements[jobname] = job_placement
                continue

            # check if num_full_nodes number of nodes are available
            if num_full_nodes > 0:
                num_checked = 0
                while num_checked < max_num_nodes and num_full_nodes > 0:
                    node_gpus = node_remaining_gpus[node_id]
                    # can take full node
                    if node_gpus == ngpus_per_node:
                        node_remaining_gpus[node_id] -= ngpus_per_node
                        num_gpus -= ngpus_per_node
                        job_placement.extend([node_id] * ngpus_per_node)
                        num_full_nodes -= 1
                    node_id = (node_id + 1) % max_num_nodes
                    num_checked += 1

            # alloc any needed gpus anywhere
            # print("num_gpu:", num_gpus)
            # print(node_remaining_gpus)
            while num_gpus > 0:
                # print("num_gpu:", num_gpus)
                if sum(node_remaining_gpus) == 0:
                    print("Error! Number of nodes is too small.")
                    exit()
                node_gpus = node_remaining_gpus[node_id]
                # print(node_id)
                # print("node_gpus: ", node_gpus)
                if node_gpus != 0:
                    can_take_gpus = min(num_gpus, node_gpus)
                    num_gpus -= can_take_gpus
                    node_remaining_gpus[node_id] -= can_take_gpus
                    job_placement.extend([node_id] * can_take_gpus)
                # advance node pointer
                node_id = (node_id + 1) % max_num_nodes
                # print("node_gpus after: ", node_gpus)
            # record placement
            placements[jobname] = job_placement
            # advance cur_node_id ptr
            cur_node_id = node_id

        return placements
