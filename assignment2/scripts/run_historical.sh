# #!/usr/bin/env bash
# # Usage: bash scripts/run_historical.sh RUN_ROOT DEVICE DB [DB ...]
# set -euo pipefail
# if [ "$#" -lt 3 ]; then
#   echo 'Usage: bash scripts/run_historical.sh RUN_ROOT DEVICE DB [DB ...]' >&2
#   exit 2
# fi
# run_root="$1"
# run_device="$2"
# shift 2
# if [ -e "$run_root" ]; then
#   echo "Output already exists: $run_root" >&2
#   exit 2
# fi
# mkdir -p "$run_root/logs"
# python run_eda.py --db "$@" --plan configs/historical.json --out "$run_root/eda" 2>&1 | tee "$run_root/logs/eda.log"
# python schema_graph.py --out "$run_root/schema" 2>&1 | tee "$run_root/logs/schema.log"
# python run_adaptive.py prepare --db "$@" --plan configs/historical.json --data "$run_root/data" 2>&1 | tee "$run_root/logs/prepare.log"
# python run_adaptive.py train --data "$run_root/data" --out "$run_root/models" --models copy7 base graph temporal additive adaptive --device "$run_device" 2>&1 | tee "$run_root/logs/train.log"
# python run_adaptive.py evaluate --data "$run_root/data" --models "adaptive" --split evaluation --out "$run_root/validation" --device "$run_device" 2>&1 | tee "$run_root/logs/validation.log"
# python run_adaptive.py evaluate --data "$run_root/data" --models "adaptive" --split test --out "$run_root/test" --device "$run_device" 2>&1 | tee "$run_root/logs/test.log"

#!/usr/bin/env bash
# Retained comparison: copy7, base, graph, temporal, additive, adaptive.
# Usage: bash scripts/run_historical.sh RUN_ROOT DEVICE DB [DB ...]
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo 'Usage: bash scripts/run_historical.sh RUN_ROOT DEVICE DB [DB ...]' >&2
  exit 2
fi
run_root="$1"
run_device="$2"
shift 2
if [ -e "$run_root" ]; then
  echo "Output already exists: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root/logs"
python run_eda.py --db "$@" --plan configs/historical.json --out "$run_root/eda" 2>&1 | tee "$run_root/logs/eda.log"
python schema_graph.py --out "$run_root/schema" 2>&1 | tee "$run_root/logs/schema.log"
python run_adaptive.py prepare --db "$@" --plan configs/historical.json --data "$run_root/data" 2>&1 | tee "$run_root/logs/prepare.log"
python run_adaptive.py train --data "$run_root/data" --out "$run_root/models" --models copy7 base graph temporal additive adaptive --device "$run_device" 2>&1 | tee "$run_root/logs/train.log"
python run_adaptive.py evaluate --data "$run_root/data" --models "$run_root/models" --split evaluation --out "$run_root/validation" --device "$run_device" 2>&1 | tee "$run_root/logs/validation.log"
python run_adaptive.py evaluate --data "$run_root/data" --models "$run_root/models" --split test --out "$run_root/test" --device "$run_device" 2>&1 | tee "$run_root/logs/test.log"