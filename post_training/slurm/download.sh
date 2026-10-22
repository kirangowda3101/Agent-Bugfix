#!/bin/bash
# Login node, once: download the pinned model and dataset into scratch, then
# prove they load with networking disabled, as they will in GPU jobs.
#   bash post_training/slurm/download.sh
source "$(dirname "$0")/common.sh"
activate_env
cd "$PT"

python -c "
from huggingface_hub import snapshot_download
from data import MODEL, MODEL_REVISION, load_split
print(snapshot_download(MODEL, revision=MODEL_REVISION))
print({s: len(load_split(s)) for s in ['train', 'validation', 'test', 'prompt']})
"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -c "
from data import load_split
from models import load_model, load_tokenizer
sizes = {s: len(load_split(s)) for s in ['train', 'validation', 'test']}
assert sizes == {'train': 374, 'validation': 90, 'test': 500}, sizes
load_tokenizer(); load_model('bfloat16')
print('offline load OK', sizes)
"
echo "DOWNLOAD OK"
