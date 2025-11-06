.PHONY: generate-aiql validate-aiql regen-aiql help health iem golden-list train activate last-eval fewshot shadow-match

PORT ?= 8000
TENANT ?= acme

generate-aiql:
	@curl -s -X POST localhost:5050/generate \
	  -H 'x-api-key: SYNAPSE@2025' -H 'content-type: application/json' \
	  -d '{"perEntity":5,"outDir":"phase3-neurons/examples/aiql","writeFiles":true,"writeCoverage":true,"strict":true}' \
	| jq '{total, written, coverageFile}'

validate-aiql:
	@node ops/validate-examples.js phase3-neurons/examples/aiql phase3-neurons/examples/aiql/_coverage.json

regen-aiql: generate-aiql validate-aiql
	@echo '[OK] regenerated + validated AIQL'

health:
	@curl -s http://localhost:$(PORT)/healthz | jq

iem:
	@curl -s -X POST http://localhost:$(PORT)/iem/build -H 'Content-Type: application/json' -d '{ "dim":256 }' | jq

golden-list:
	@curl -s -H 'x-tenant-id: $(TENANT)' http://localhost:$(PORT)/golden/list | jq

train:
	@curl -s -X POST http://localhost:$(PORT)/trainer/run -H 'Content-Type: application/json' -H 'x-tenant-id: $(TENANT)' -d '{ "tenant":"$(TENANT)","maxGrid":64,"maxFewshot":2000 }' | jq

activate:
	@[ -z "$(CKPT)" ] && echo "CKPT=trainer_YYYYMMDD_HHMM required" && exit 1 || true
	@curl -s -X POST http://localhost:$(PORT)/trainer/activate -H 'Content-Type: application/json' -d '{ "checkpoint":"$(CKPT)" }' | jq

last-eval:
	@curl -s http://localhost:$(PORT)/trainer/last_eval | jq

fewshot:
	@curl -s 'http://localhost:$(PORT)/fewshot/show?tenant=$(TENANT)' | jq

shadow-match:
	@curl -s -X POST http://localhost:$(PORT)/synapse/match \
	 -H 'Content-Type: application/json' -H 'x-tenant-id: $(TENANT)' -H 'x-experiment: train-shadow' \
	 -d '{ "intent": { "ask":"top_k", "metric":{"op":"sum","target":"orders.total_amount"},"target":"orders.shipping_city"} }' | jq

help:
	@echo 'AIQL Auto-Question Generator Commands:'
	@echo '  make generate-aiql  - Generate AIQL examples'
	@echo '  make validate-aiql  - Validate AIQL examples'
	@echo '  make regen-aiql     - Generate + validate'
	@echo ''
	@echo 'Phase3 Neurons Commands:'
	@echo '  make health         - Health check'
	@echo '  make iem            - Build IEM'
	@echo '  make golden-list    - List golden testcases (TENANT=acme)'
	@echo '  make train          - Run trainer (TENANT=acme)'
	@echo '  make activate CKPT=... - Activate checkpoint'
	@echo '  make last-eval      - Get last eval results'
	@echo '  make fewshot         - Show few-shot config (TENANT=acme)'
	@echo '  make shadow-match   - Run shadow match comparison (TENANT=acme)'
	@echo ''
	@echo '  make feedback-demo  - Seed feedback data'
	@echo '  make eval           - Get last eval results (legacy)'
	@echo '  make trainer-run    - Run trainer (legacy)'
	@echo '  make trainer-activate CKPT=... - Activate checkpoint (legacy)'
	@echo ''
	@echo 'Variables:'
	@echo '  PORT=8000           - Server port (default: 8000)'
	@echo '  TENANT=acme         - Tenant ID (default: acme)'

.PHONY: feedback-demo eval trainer-run trainer-activate

feedback-demo:
	@cd phase3-neurons && python3 -c "import os, json, time, uuid; root = os.getcwd(); path = os.path.join(root,'data','feedback.jsonl'); os.makedirs(os.path.dirname(path),exist_ok=True); [open(path,'a').write(json.dumps({'id': str(uuid.uuid4()), 'ts': int(time.time()*1000), 'tenant':'default', 'intent': {'ask':'top_k','metric':{'op':'sum'}}, 'chosenTarget': 'orders.total_amount', 'otherSlots': {'category':'orders.status','timestamp':'orders.created_at'}, 'iemHash': 'demo', 'latencyMs': 100+i})+'\n') for i in range(20)]; print('seeded ->', path)"

eval:
	@curl -s localhost:5050/trainer/last_eval -H 'x-api-key: SYNAPSE@2025' | jq '.'

trainer-run:
	@curl -s -X POST localhost:5050/trainer/run -H 'x-api-key: SYNAPSE@2025' -H 'content-type: application/json' -d '{}' | jq '.'

trainer-activate:
	@[ -n "$(CKPT)" ] || (echo "CKPT required, e.g. make trainer-activate CKPT=trainer_20251028_1405" && exit 1)
	@curl -s -X POST localhost:5050/trainer/activate -H 'x-api-key: SYNAPSE@2025' -H 'content-type: application/json' -d '{"checkpoint":"$(CKPT)"}' | jq '.'

