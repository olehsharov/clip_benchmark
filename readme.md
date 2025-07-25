### Prerequisites
Ubuntu 20.04 / 22.04, CUDA 12, cudNN

### All in one script
```bash
./run.sh # creates venv, install deps, generate test data, run tests
```

If for some reason it does not work do this
### 1. Create env
`python3.10 -m venv .venv`

### 2. Activate env
`source .venv/bin/activate`

### 3. Install deps
`pip install -r reuirements.txt`

### 4. Generate test data
`python main.py generate --count 10000`

### 5. Test FPS
```bash
python main.py test --workers=<num_workers> --batch_size=<batch_size>
# num_workers - how many models load in single gpu; default 8
# batch size per model; depends on VRAM; default 420
```
