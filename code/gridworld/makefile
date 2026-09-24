PYTHON ?= python
TASK_ID ?=
TASK_ARGS = $(if $(TASK_ID),--task-id $(TASK_ID),)

.PHONY: ablation_unconstrained ablation_penalty ablation_penalty_smaller_costs ablation_VRPDPG ablation_PDPG ablation_norm_constraint

ablation_unconstrained:
	$(PYTHON) Penalty_run.py 1_UnconAblation.json $(TASK_ARGS)

ablation_penalty:
	$(PYTHON) Penalty_run.py 2_PenaltyAblation.json $(TASK_ARGS)

ablation_penalty_smaller_costs:
	$(PYTHON) Penalty_run.py 3_CostAblation.json $(TASK_ARGS)

ablation_VRPDPG:
	$(PYTHON) VRPDPG_run.py 4_VRPDPG.json $(TASK_ARGS)

ablation_PDPG:
	$(PYTHON) PDPG_run.py 5_PDPG.json $(TASK_ARGS)

ablation_norm_constraint:
	$(PYTHON) Norm_constr_run.py 6_Norm_constr.json $(TASK_ARGS)
