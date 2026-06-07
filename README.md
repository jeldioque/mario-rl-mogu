# 🍄 Team MOGU — Super Mario Bros Deep RL

**DL2026 Final Project** — Training a PPO agent to play Super Mario Bros using deep reinforcement learning.

**Headline result: PPO achieved 1810 mean episode reward and 29/100 level completions — a 74% improvement over the DQN baseline (1038 reward, 0 completions).**

---

## Reproduce in 5 steps

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/mario-rl-mogu
cd mario-rl-mogu
```

**2. Install system dependencies (WSL/Ubuntu)**
```bash
sudo apt install -y ffmpeg swig libgl1 libglib2.0-0
```

**3. Install Python dependencies**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**4. Run evaluation** — reproduces headline number from checkpoint
```bash
python evaluate.py
```

Expected output:
```
  Mean Reward     : 1810.0 ± 280.0
  Mean X-Position : 1440.0
  Completions     : 29/100 (29.0%)
```

**5. Watch the agent play**
```bash
python -c "
from stable_baselines3 import PPO
from src.wrappers import make_eval_env
import numpy as np

model = PPO.load('checkpoints/best.pt')
env = make_eval_env()
obs, _ = env.reset()
done, total = False, 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, terminated, truncated, info = env.step(int(action))
    total += r
    done = terminated or truncated
print(f'Reward: {total:.1f} | X-pos: {info.get(\"x_pos\", 0)}')
env.close()
"
```

---

## Results

| Model | Mean Reward | X-Position | Completions | Training Steps |
|-------|------------|------------|-------------|----------------|
| DQN (Baseline) | 1038 ± 0 | 1127 | 0/100 | 500k |
| **PPO (Ours)** | **1810 ± 280** | **1440** | **29/100** | **1.35M** |

---

## Project Structure

```
mario-rl-mogu/
├── train.py              # Entry point — trains DQN or PPO
├── evaluate.py           # Loads checkpoints/best.pt, prints headline
├── configs/
│   └── default.yaml      # All hyperparameters
├── src/
│   ├── model.py          # CNN architectures + agent builders
│   ├── wrappers.py       # Environment preprocessing pipeline
│   ├── utils.py          # Config loading, metric saving
│   └── watch_agent.py    # Record the model result 
├── notebooks/
│   ├── 01-eda.ipynb      # Environment exploration
│   ├── 02-train.ipynb    # Training documentation
│   └── 03-ablations.ipynb# Ablation study
├── checkpoints/
│   └── best.pt           # Best trained PPO model
└── results/
    ├── metrics.json      # All evaluation numbers
    └── figures/          # Plots for report
```

---

## Reproducibility

- **Hardware**: NVIDIA RTX 4050 Laptop GPU (6GB VRAM), WSL2 Ubuntu 24
- **Training time**: ~3 hours (PPO, 1.35M steps, 4 parallel envs)
- **Seed**: 42
- **Python**: 3.12.3
- **Key packages**: stable-baselines3==2.3.2, torch==2.2.2+cu121, gym-super-mario-bros==7.4.0

---

## License

MIT License — see [LICENSE](LICENSE)
