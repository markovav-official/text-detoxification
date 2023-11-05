# Text Detoxification Project

## Project Description

This project aims to identify and mitigate toxic comments in text data, contributing to a healthier online communication environment. It includes a baseline approach based on word filtering and an advanced method using the T5-Small transformer model for natural language understanding.

## Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
- [Reports](#Reports)

## Installation

To set up the project environment:

```bash
git clone https://github.com/markovav-official/text-detoxification.git
cd text-detoxification
pip install -r requirements.txt
```

Note: this repository uses Git LargeFileStorage ([git lfs](https://git-lfs.com/)) for storing models and datasets.

## Usage

```bash
python3 src/models/detoxify.py "text"
```

If you want to use other model or run this file from different directory:

```bash
python3 src/models/detoxify.py "text" "model_path"
```

## Reports

- [Solution building](reports/solution-building.md)
- [Final report](reports/final-report.md)
