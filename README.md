<div align="center">
  <h2><b>(NeurIPS '25) MIRA: Medical Time Series Foundation Model for Real-World Health Data </b></h2>
</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2506.07584">Paper Page</a>]**

</div>

## Overview

MIRA unifies representation learning across multiple medical time-series datasets and supports zero-shot forecasting for downstream clinical prediction tasks.

**Key features**
- CT-RoPE for continuous-time positional encoding
- Frequency-specialized MoE to adapt across rhythms
- Neural-ODE extrapolation for forecasting at arbitrary timestamps

<p align="center">
  <img src="images/Model_Architecture.png" width="720"/>
</p>

---

## Installation

Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
pip install torchdiffeq
```

**Note: MIRA requires `transformers==4.40.1` .**

---
## Data Preparation

### Data format example

Each line represents one sample and must contain at least `sequence` and
`time` fields:

``` json
{"sequence": [1.0, 1.2, 0.8, ...], "time": [0.12, 1.52, 2.31, ...], "mask": [1,1,1,...]}
{"sequence": [5.1, 5.0, 5.3, ...], "time": [1699990000, 1699990600, 1699991200, ...], "mask": [1,1,1,...]}
```
---
## Training

For distributed training on irregular medical data:

```bash
python torch_dist_run.py main.py \
  --from_scratch \
  -d ./yourdata.jsonl \
  --output_path ./saveyoucheckpoints \
  --save_steps 10000 \
  --save_strategy steps \
  --save_total_limit 10 \
  --save_only_model \
  --precision bf16 \
  --time_aware_dataset \
  --time_normalization_method minmax
```

For full argument list:

```bash
python main.py --help
```


## Inference

```python
import torch
from transformers import AutoModelForCausalLM

context_length = 12
normed_seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

model = AutoModelForCausalLM.from_pretrained(
    'MIRA',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]
```

## Citation

> Please let us know if you find out a mistake or have any suggestions!

> If you find the MIRA models helpful in your research, please consider to star this repository and cite the
> corresponding [paper](https://arxiv.org/abs/2506.07584):

```
@article{li2025mira,
  title={MIRA: Medical Time Series Foundation Model for Real-World Health Data},
  author={Li, Hao and Deng, Bowen and Xu, Chang and Feng, Zhiyuan and Schlegel, Viktor and Huang, Yu-Hao and Sun, Yizheng and Sun, Jingyuan and Yang, Kailai and Yu, Yiyao and others},
  journal={arXiv preprint arXiv:2506.07584},
  year={2025}
}
```

# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
