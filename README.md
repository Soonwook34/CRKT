# CRKT
This is our official implementation for the paper Enhancing knowledge tracing with concept map and response disentanglement (Knowledge-Based Systems).

<img alt="CRKT" src="assets/CRKT_architecture.jpg">

## Experiment Environment
- python 3.10+
- torch 2.0+
- torch_geometric 2.4+
- scikit-learn 1.4+
- pandas 2.2.0+
- tqdm, 

### Environment Setting Example
```bash
pip3 install torch torchvision torchaudio
pip3 install torch_geometric
pip3 install scikit-learn pandas tqdm
```

## Basic Usage
### Preprocessing Dataset
```python
python3 data_preprocess.py --dataset DBE_KT22
```

### Run Experiment
```python
python3 main.py --dataset DBE_KT22 --model CRKT --batch 128 --lr 1e-3 --dim_c 32 --dim_q 32 --dim_g 32 --lamb 0.1 --alpha 0.1 --top_k 10 --beta 0.01 --exp_name "test"
```
