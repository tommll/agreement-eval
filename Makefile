.PHONY: install db-up db-down db-reset initdb demo html test sroie local funsd clean

VENV ?= .venv
PY   ?= $(VENV)/bin/python
CLI  ?= $(VENV)/bin/agreement-eval

install:                      ## create the venv and install the package
	uv venv $(VENV) --python 3.10
	uv pip install --python $(VENV)/bin/python -e ".[dev]"

db-up:                        ## start Postgres
	docker compose up -d
	@until docker exec agreement-eval-pg pg_isready -U agreement >/dev/null 2>&1; do sleep 1; done
	@echo "postgres ready on localhost:5432"

db-down:
	docker compose down

db-reset:                     ## destroy the database volume and recreate it
	docker compose down -v
	$(MAKE) db-up initdb

initdb: ; $(CLI) initdb

demo: db-up initdb            ## offline end-to-end run: synthetic receipts, mock models
	$(CLI) demo

sroie: db-up initdb           ## real run: SROIE receipts, Claude configs (costs money)
	$(CLI) all --experiment experiments/sroie.yaml

local: db-up initdb           ## real run on LOCAL models (start scripts/serve_local.sh first)
	$(CLI) all --experiment experiments/sroie-local.yaml

funsd: db-up initdb           ## real run: FUNSD forms (needs the dataset downloaded first)
	$(CLI) all --experiment experiments/funsd.yaml

EXP ?= experiments/sroie-local.yaml

html:                         ## export the latest run to report.html and open it
	$(CLI) evaluate --experiment $(EXP) --html --open --quiet

test: ; $(PY) -m pytest tests -q

clean:
	rm -rf reports/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
