# STONE-SEE
STONE-SEE is a deep learning-powered system designed to automatically detect kidney stones from CT scan images, delivering accurate and efficient diagnostic support for healthcare professionals

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EfkCQ_cby85FhOWLKgL_J-sBofEwZ_Skg1w1hibxwLaKYA) | [Slack channel](https://app.slack.com/client/T09CPAEDU21/C09F1MB238V) | [Project report](https://www.overleaf.com/project/68c647009c2695d04735951f) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

## Dataset

The dataset for this project is hosted on **Google Drive**. You can download the files from the following link:

[Download the dataset from Google Drive](https://drive.google.com/drive/folders/1_Y6ztdLwIZyF-MBIQhiXoS-TEXul6EIK?usp=drive_link)


### What to find where

Explain briefly what files are found where

```bash
repository
├── data                          ## Contains all data-related files
│   ├── raw/                      ## Raw data: original CT scans
│   │   ├── stone/                ## CT scans with kidney stones
│   │   └── no_stone/             ## CT scans without kidney stones
│   ├── augmented/                ## Augmented data (e.g., rotated, flipped)
│   │   ├── stone/                ## Augmented images of kidney stones
│   │   └── no_stone/             ## Augmented images without kidney stones
│   ├── processed/                ## Preprocessed images (resized, normalized)
│   └── augmented_processed/      ## Augmented and preprocessed images
├── src/                          ## Source code for your project
│   ├── preprocessing/            ## Data preprocessing scripts (e.g., resize, normalize)
│   ├── models/                   ## Model architectures and training scripts (CNN, etc.)
│   └── evaluation/               ## Evaluation scripts (e.g., accuracy, precision)
├── notebooks/                    ## Jupyter notebooks for exploration and training
├── results/                      ## Output files
│   ├── figures/                  ## Plots and graphs (e.g., training curves, confusion matrices)
│   ├── models/                   ## Saved models (e.g., trained model weights)
│   └── logs/                     ## Training logs, evaluation logs
├── docs/                         ## Documentation files (e.g., reports)
├── README.md                     ## Project overview, setup instructions, and more
├── requirements.yml              ## Dependencies for the project (if using conda)
└── .gitignore                    ## List of files to ignore in Git (e.g., .DS_Store)
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
# Download dataset from Google Drive and place in data/ directory
# Dataset structure: data/Augmented_Dataset/ and data/Original_Dataset/

# Activate conda environment
conda activate kidneynet

# Train the model
python src/models/train_model.py

# Evaluate the model
python src/evaluation/evaluate_model.py

# Compare datasets (optional)
python src/evaluation/compare_datasets.py

# Generate evaluation report
python src/evaluation/generate_report.py

# Run the web application with the trained model
python run.py
```
Data can be found at [Google Drive](https://drive.google.com/drive/folders/1_Y6ztdLwIZyF-MBIQhiXoS-TEXul6EIK?usp=drive_link). Download and extract to the `data/` directory with subdirectories `Augmented_Dataset/` and `Original_Dataset/`, each containing `Stone/` and `Non-Stone/` folders.
Output will be saved in `results/` directory: trained models in `results/models/`, evaluation metrics in `results/evaluation_metrics.json`, and visualizations in `results/figures/`. After training, you can run `python run.py` to start the web application which will use the trained model for predictions.

## 5. Running the Web Application (Cross-Platform)

The project includes a cross-platform runner that works on **Windows, macOS, and Linux**.

### Prerequisites
- Python 3.7 or higher
- Node.js and npm (for frontend)
- All Python dependencies installed (`pip install -r requirements.txt`)

### Quick Start

**Windows:**
```batch
run.bat
```
Or:
```batch
python run.py
```

**macOS/Linux:**
```bash
python3 run.py
```
Or make it executable:
```bash
chmod +x run.py
./run.py
```

### What the Runner Does

1. **Checks dependencies** - Verifies Python, Node.js, and required files
2. **Stops existing servers** - Kills any processes on ports 5000 and 3000
3. **Starts backend** - Launches Flask API server on http://localhost:5000
4. **Starts frontend** - Launches Next.js dev server on http://localhost:3000
5. **Opens browser** - Automatically opens the web interface
6. **Handles cleanup** - Properly stops servers when you press Ctrl+C

### Manual Start (Alternative)

If you prefer to start servers manually:

**Backend:**
```bash
cd backend
python app.py
```

**Frontend (in a new terminal):**
```bash
cd frontend
npm install  # First time only
npm run dev
```

### Troubleshooting

- **Port already in use**: The runner will try to kill existing processes, but you may need to manually stop them
- **Backend fails to start**: Check `logs/backend.log` and ensure all Python dependencies are installed
- **Frontend fails to start**: Check `logs/frontend.log` and ensure Node.js is installed and `npm install` has been run
- **Model not found**: Make sure you've trained the model first (see training instructions)

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
