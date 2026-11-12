# Examples

| file | what it shows |
|---|---|
| [github-actions.yml](github-actions.yml) | checking a deployed endpoint from CI with a committed reference file |
| [probes.jsonl](probes.jsonl) | a custom probe set (`embed-parity reference --probes probes.jsonl ...`) |
| [check_in_python.py](check_in_python.py) | calling the checker from Python and reading the result as data |
| [transformers_server.py](transformers_server.py) | a small real-model server with switchable bugs, for trying the tool on CPU |

## Trying it on a real model without a GPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "embed-parity[reference] @ git+https://github.com/cb-Sigma/embed-parity"

embed-parity reference --model sentence-transformers/all-MiniLM-L6-v2 --out minilm.ref.json

# terminal 1: a server with the wrong pooling
python examples/transformers_server.py --pooling cls

# terminal 2
embed-parity check --base-url http://127.0.0.1:8088 --reference minilm.ref.json
```

Other switches: `--no-normalize`, `--ignore-mask`, `--max-length 128`, and for rerank
`--rerank-model cross-encoder/ms-marco-MiniLM-L6-v2 [--rerank-raw]`.

## Trying it with no model at all

```bash
embed-parity serve-sim --fault "drop-prompt,dim=32"      # terminal 1
embed-parity check --base-url http://127.0.0.1:8765 --model-id toy-embed   # terminal 2
embed-parity serve-sim --list-faults
```
