# Convenience targets -- each is a thin wrapper over a script that can
# also be run directly (scripts are the source of truth, not this file).

.PHONY: verify-fresh sync-check pr1-gate integration lint

# Full CPU verification against an installed vllm==0.24.0 (no GPU).
verify-fresh:
	bash scripts/verify_fresh_clone.sh

# PR1/PR2 lightning drift gate (pure stdlib, runs anywhere).
sync-check:
	python3 scripts/check_pr1_pr2_lightning_sync.py

# PR1-only Docker gate (22 tests + ruff), matches CI.
pr1-gate:
	docker run --rm -v "$(CURDIR):/deliverable/minicpm_sala_stage1_pr" \
		nvidia/cuda:12.4.1-devel-ubuntu22.04 \
		bash /deliverable/minicpm_sala_stage1_pr/docker_run_pr1.sh

# Full-stack Docker gate (PR1 + PR2 overlay).
integration:
	docker run --rm -v "$(CURDIR):/deliverable/minicpm_sala_stage1_pr" \
		nvidia/cuda:12.4.1-devel-ubuntu22.04 \
		bash /deliverable/minicpm_sala_stage1_pr/docker_run_integration.sh

lint:
	ruff check vllm/ pr2/vllm/ && ruff format --check vllm/ pr2/vllm/
