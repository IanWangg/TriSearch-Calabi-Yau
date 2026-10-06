"""Opt-in MOSEK thread allocation for concurrent geometry workers."""

import os

if "CY_EVAL_MOSEK_NUM_THREADS" in os.environ:
    import mosek

    thread_count = int(os.environ["CY_EVAL_MOSEK_NUM_THREADS"])
    if thread_count <= 0:
        raise ValueError("CY_EVAL_MOSEK_NUM_THREADS must be positive.")
    original_task_init = mosek.Task.__init__

    def initialize_task(self, *args, **kwargs):
        original_task_init(self, *args, **kwargs)
        self.putintparam(mosek.iparam.num_threads, thread_count)

    mosek.Task.__init__ = initialize_task
