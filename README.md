# DynamicDTA
DynamicDTA: Drug-Target Binding Affinity Prediction Using Dynamic Descriptors and Graph Representation


## Methods: 

![framework](/framework.png)

DynamicDTA
integrates protein dynamics descriptors, which are derived from molecular dynamics simulations, offering a more comprehensive understanding of protein behavior. 

## Running DynamicDTA

### Setting Up the Environment
We provide a conda environment configuration file to easily set up the required Python environment.
1. Download or clone the repository:

   ```bash
   git clone https://github.com/shmily-ld/DynamicDTA.git
   cd DynamicDTA
   ```

2. Create the environment using the provided `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the conda environment:

   ```bash
   conda activate ld
   ```
4. Install RDKit:

   ```bash
   pip install rdkit
   ```
### Dataset Preparation

```bash
python data_preprocessing.py
```

### Training and Evaluation

```bash
python training.py
```

### Visualization

```bash
python visualization.py
```

![PixPin_2025-01-21_16-20-44](./visualization.png)


### Cite
```bibtex
@article{luo2025dynamicdta,
  title={DynamicDTA: Drug-Target Binding Affinity Prediction Using Dynamic Descriptors and Graph Representation},
  author={Luo, Dan and Zhou, Jinyu and Xu, Le and Yuan, Sisi and Lin, Xuan},
  journal={arXiv preprint arXiv:2505.11529},
  year={2025}
}
