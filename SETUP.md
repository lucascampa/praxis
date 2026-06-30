# Setup Instructions

Steps for running RESPLIT vs. PRAXIS comparison on the airline satisfaction dataset.

**This project extends the original DIMEX work** — see [SPLIT context.md](SPLIT%20context.md) for background.

- **Windows** users must enable WSL (step #1)
- **Mac/Linux** → skip step #1
- **All platforms**: WSL or native bash/zsh works fine



## 1. WSL (Windows Subsystem for Linux)
Control Panel → Turn Windows features on or off → Make sure Windows Subsystem for Linux is enabled (may need to restart device) → Search `wsl` and launch it

## 2. Miniconda
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh
bash Miniconda3-latest-Linux-x86\_64.sh
source ~/.bashrc
```

## 3. Clone this Repository

**If you already have the repo locally:**
```bash
cd /path/to/praxis
```

**If not:**
```bash
cd "/mnt/c/where/you/want/to/clone/this/repo/to"
git clone https://github.com/your-username/praxis.git
cd praxis
```

## 4. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate praxis-env
```

## 5. Install System Dependencies (Linux/WSL only)
```bash
conda install -c conda-forge libgcc-ng libstdcxx-ng
sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y libtbb-dev pkg-config
sudo apt install -y libgmp-dev
```

## 6. Install SPLIT and RESPLIT
```bash
pip install --upgrade pip
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
cd SPLIT-ICML
pip install resplit/ split/
cd ..
```

**Note**: This installs both SPLIT (branch-and-bound single tree) and RESPLIT (Rashomon set enumeration).

## 7. Install PRAXIS (via PyPI)
PRAXIS is installed automatically via `environment.yml` (pip section). If needed manually:
```bash
pip install tree-praxis
```

## ⚠️ Important: RESPLIT Execution Constraint
**RESPLIT can only be run via command-line scripts, NOT Jupyter notebooks.**

According to the SPLIT-ICML README:
- RESPLIT has known timeout issues in Jupyter
- Run RESPLIT via `python run_script.py` or SLURM scripts
- Use Jupyter only for PRAXIS comparisons and analysis

Example workflow:
```bash
# Run RESPLIT via command-line script
python resplit_runner.py

# Then analyze results in Jupyter if needed
jupyter notebook notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb
```

## 8. Install the `dimex` Package (local utilities)
From the praxis repo root:
```bash
pip install -e .
```

## 9. Verify Installations
```bash
python -c "import split; print('✓ SPLIT installed')"
python -c "from praxis import PRAXIS; print('✓ PRAXIS installed')"
python -c "import dimex; print('✓ dimex package installed')"
```

All three should print success messages.

## 10. Run the Notebooks

**Option A: Baseline DIMEX work** (SPLIT vs. XGBoost)
```bash
jupyter notebook notebooks/Black-Box\ to\ Glass-Box\ modeling\ \[RANDOM_SEED=42\].ipynb
```

**Option B: RESPLIT vs. PRAXIS comparison** (NEW - after creating notebook)
```bash
jupyter notebook notebooks/RESPLIT_vs_PRAXIS_comparison.ipynb
```

