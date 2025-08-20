# The Prompt Stealing Fallacy: Rethinking Metrics, Attacks, and Defenses

## Overview

This repository contains the implementation of our research on prompt stealing attacks and defenses. We present one novel attack method and two defense strategies to address the vulnerability of prompt stealing in T2I Models.

## Project Structure

```
pthieft/
├── attack/                 # Attack implementation
│   └── ms-swift/          # The RL framework we used
├── defense/                # Defense implementations
│   └── psa-defense-main/  # PSA defense framework
└── README.md              # This file
```

## Attack Method

### Attack
Located in `./attack/`

Our attack leverages the MS-SWIFT framework to perform prompt stealing attacks on T2I models. The attack method is designed to extract sensitive prompts.


## Defense Methods

### PSA Defense Framework
Located in `/fred/oz339/zdeng/pthieft/defense/psa-defense-main/`

We implement two defense strategies:

#### 1. White-Box Defense (`wb`)
The white-box defense provides stronger protection by having access to model internals and gradients.

**Run the defense:**
```bash
cd defense/psa-defense-main/
bash scripts/run_wb.sh
```

**Key components:**
- Adversarial example generation with white-box access


#### 2. Black-Box Defense (`bb`)
The black-box defense operates without access to model internals, making it more practical for real-world deployment.

**Run the defense:**
```bash
cd defense/psa-defense-main/
bash scripts/run_bb.sh
```

## Environment Setup

All required dependencies are listed in the respective `requirements.txt` files in each subdirectory.

### Installation

It is recommended to use a virtual environment (e.g., `virtualenv` or `conda`) to manage the project dependencies.

To install the required packages, run:

```bash
pip install -r requirements.txt
```

