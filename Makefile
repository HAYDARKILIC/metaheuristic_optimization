.PHONY: install test notebooks solutions check clean

install:
	pip install -r requirements.txt pytest

test:
	pytest -q tests

# Re-execute every notebook in place (regenerates outputs and figures/).
notebooks:
	for nb in week*/*.ipynb; do \
		echo "== $$nb"; \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$$nb" || exit 1; \
	done

# Run every code block in solutions/*.md standalone.
solutions:
	python tools/run_solution_blocks.py

# Fail if any committed notebook contains a failed validation.
check:
	@! grep -l '\[FAIL\]' week*/*.ipynb && echo "no failed validations"

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +
