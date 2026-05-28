
## Knowledge:
Real benchmark evaluation should live in run-local `.autopilot/eval_programs/` workspaces.  The teacher can plan external repos, write evaluator code, refine parsers, and run smoke checks there.  Do not treat smoke eval as a stop condition.



## Knowledge:
Autonomous training runs must treat processes as first-class resources. Every long-running service or training job should be started through the process registry, with PID, process_group_id, command, environment, log_file, and pid_file recorded under `.autopilot/processes/`. The teacher agent may use `process_list`/`process_kill` or the `kill_process` action to free GPUs before vLLM, LLaMA-Factory, benchmark harnesses, or other expensive jobs. Real benchmark evaluation must also be agent-owned. Before trusting a score, the agent should create or select a run-local evaluator in `.autopilot/eval_programs/`, pin external repos/commands when needed, write generated evaluator code only in that sandbox, smoke-test it, refine it after failures, and only then use `run_eval` results for target stopping.