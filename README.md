<h1 align="center">TerminalTraj</h1>

<div align="center">
🤗 <a href="https://huggingface.co/m-a-p/TerminalTraj-32B"><b>Model</b></a>&nbsp&nbsp | &nbsp&nbsp
🤗 <a href="https://huggingface.co/datasets/m-a-p/TerminalTraj"><b>Data</b></a>&nbsp&nbsp | &nbsp&nbsp
🤗 <a href="https://arxiv.org/abs/2602.01244"><b>HF&nbsp&nbspPaper</b></a>&nbsp&nbsp | &nbsp&nbsp
📑 <a href=""><b>arXiv</b></a>&nbsp&nbsp
</div>


This is the repo for the paper **TerminalTraj: Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments**

## Brief
Training agentic models for terminal-based tasks critically depends on high-quality terminal trajectories that capture realistic long-horizon interactions across diverse domains. However, constructing such data at scale remains challenging due to two key requirements: **_Executability_**, since each instance requires a suitable and often distinct Docker environment; and **_Verifiability_**, because heterogeneous task outputs preclude unified, standardized verification.

To address these challenges, we propose **TerminalTraj**, a scalable pipeline that (i) filters high-quality repositories to construct Dockerized execution environments, (ii) generates Docker-aligned task instances, and (iii) synthesizes agent trajectories with executable validation code. Using TerminalTraj, we curate 32K Docker images and generate **50,733** verified terminal trajectories across eight domains.

Models trained on this data with the Qwen2.5-Coder backbone achieve consistent performance improvements on TerminalBench (TB), with gains of up to **20%** on TB 1.0 and **10%** on TB 2.0 over their respective backbones. Notably, **TerminalTraj-32B** achieves strong performance among models with fewer than 100B parameters, reaching **35.30%** on TB 1.0 and **22.00%** on TB 2.0, and demonstrates improved test-time scaling behavior.

## Method

We propose **TerminalTraj**, a large-scale pipeline for generating Docker-aligned terminal agent trajectories from real-world GitHub repositories, with instance-specific executable validation.

To scale environments beyond heuristic repository filtering, we cast repository selection as model-based quality scoring, enabling automated construction of **32,325** Docker images across eight programming languages. We further curate instances spanning **eight** specialized domains with real-world tools and dependencies.

TerminalTraj filters rollouts via task-specific executable validators (inspired by TerminalBench). Overall, TerminalTraj produces **50,733** verified trajectories and supports continual, scalable data synthesis.



<p align="center">
  <img src="https://github.com/user-attachments/assets/a036976d-d6be-4e8d-93a9-36ed91e5abf1" width="600" alt="image" />
</p>

## Results


<p align="center">
  <img src="https://github.com/user-attachments/assets/a8cb8eb2-cab8-40c6-b133-4a95a97662ee" width="400" alt="image" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b1134f1-9f80-49a5-af57-0a1e89d8dbd5" width="700" alt="image" />
</p>


As shown in the figure above, **TerminalTraj-32B** achieves state-of-the-art performance among models under 100B parameters on both **TB1.0** and **TB2.0**, and its performance is close to **Qwen3-Coder-480B**.

In addition, we find that **TerminalTraj**, with its large-scale agentic data grounded in real-world environments, can substantially enhance a model’s **test-time scaling** capability.


## Dataset

We have released our training dataset (trajectory):

```python
from datasets import load_dataset

ds = load_dataset("m-a-p/TerminalTraj")
```

## Models

We have released our TerminalTraj-7B/14B/32B models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "m-a-p/TerminalTraj-14B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # 14B建议用fp16或bf16
    device_map="auto"            # 自动分配GPU
)
```



## TODO

We will also release an additional 5,000 instances equipped with Docker-based environments in the near future.

## Citation

**BibTeX:**

```bibtex
@misc{wu2026largescaleterminalagentictrajectory,
      title={Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments}, 
      author={Siwei Wu and Yizhi Li and Yuyang Song and Wei Zhang and Yang Wang and Riza Batista-Navarro and Xian Yang and Mingjie Tang and Bryan Dai and Jian Yang and Chenghua Lin},
      year={2026},
      eprint={2602.01244},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.01244}, 
}
```
