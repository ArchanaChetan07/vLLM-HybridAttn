.PHONY: init init-force health overlay test-cpu lint fmt \
	gate-quick gate-pr1 gate-full gpu-validation install-infllm verify-fresh

REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

init:
	bash scripts/dev/init-dev-env.sh

init-force:
	bash scripts/dev/init-dev-env.sh --force

health:
	bash scripts/dev/health-check.sh

overlay:
	bash scripts/install_pr2_overlay.sh

gate-quick:
	bash scripts/dev/run-gates.sh quick

gate-pr1:
	bash scripts/dev/run-gates.sh pr1

gate-full:
	bash scripts/dev/run-gates.sh full

test-cpu:
	rm -rf /tmp/minicpm_tests && mkdir -p /tmp/minicpm_tests/v1/core \
		/tmp/minicpm_tests/v1/attention /tmp/minicpm_tests/models/language/generation
	cp tests/models/language/generation/test_minicpm_sala_*.py \
		/tmp/minicpm_tests/models/language/generation/
	cp pr2/tests/v1/core/test_minicpm_sala_*.py /tmp/minicpm_tests/v1/core/
	cp pr2/tests/v1/attention/test_minicpm_sala_*.py /tmp/minicpm_tests/v1/attention/
	cd /tmp && python3 -m pytest --noconftest --rootdir=/tmp/minicpm_tests \
		/tmp/minicpm_tests/models/language/generation/ \
		/tmp/minicpm_tests/v1/core/ /tmp/minicpm_tests/v1/attention/ -q

lint:
	ruff check vllm/model_executor/models/minicpm_sala.py pr2/vllm

fmt:
	ruff format vllm/model_executor/models/minicpm_sala.py pr2/vllm

gpu-validation:
	bash pr2/scripts/gpu_validation/run_all_gpu_validation.sh

verify-fresh:
	bash scripts/verify_fresh_clone.sh

install-infllm:
	bash scripts/install_infllm_v2.sh