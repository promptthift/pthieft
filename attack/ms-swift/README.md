# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

## 🛠️ Installation
To install using pip:
```shell
pip install ms-swift -U
```

To install from source:
```shell
# pip install git+https://github.com/modelscope/ms-swift.git

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

Running Environment:

|              | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python       | >=3.9        | 3.10        |                                           |
| cuda         |              | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        |             |                                           |
| transformers | >=4.33       | 4.51      |                                           |
| modelscope   | >=1.19       |             |                                           |
| peft | >=0.11,<0.16 | ||
| trl | >=0.13,<0.17 | 0.16 |RLHF|
| deepspeed    | >=0.14       | 0.14.5 | Training                                  |
| vllm         | >=0.5.1      | 0.7.3/0.8.3       | Inference/Deployment/Evaluation           |
| lmdeploy     | >=0.5        | 0.7.2.post1       | Inference/Deployment/Evaluation           |
| evalscope | >=0.11       |  | Evaluation |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).
