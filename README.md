# SFU CMPT 340 Kidney Net

This repository is for the **CMPT 340 course project**. The **Kidney-Net** project focuses on detecting **kidney stones** from **CT scan images** using deep learning techniques. The project includes a **convolutional neural network (CNN)** model that classifies CT scan images into two categories: **kidney stone** and **no kidney stone**. The dataset includes both **original** and **augmented** images to improve the model's accuracy and robustness.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EfkCQ_cby85FhOWLKgL_J-sBofEwZ_Skg1w1hibxwLaKYA) | [Slack channel](https://app.slack.com/client/T09CPAEDU21/C09F1MB238V) | [Project report](https://www.overleaf.com/project/68c647009c2695d04735951f) |
|-----------|---------------|-------------------------|
| Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project. | Slack channel: Link your private Slack project channel. | Project report: Link your Overleaf project report document. |

## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

## Table of Contents
1. [Demo](#demo)
2. [Installation](#installation)
3. [Reproducing this project](#repro)
4. [Guidance](#guide)

<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work:

```python
from kidney_net import kidneyexample
imgs = kidneyexample.demo()
for img in imgs:
    view(img)
What to find where
This section describes where specific files and folders are located in the project structure:

bash
Copy code
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
<a name="installation"></a>

2. Installation
Provide sufficient instructions to reproduce and install your project.
Provide exact versions, test on CSIL or reference workstations.

bash
Copy code
git clone https://github.com/your-username/Kidney_net.git
cd Kidney_net
conda env create -f requirements.yml
conda activate kidney-net
<a name="repro"></a>

3. Reproduction
Demonstrate how your work can be reproduced, e.g., the results in your report.

bash
Copy code
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate kidney-net
python evaluate.py --epochs=10 --data=/input/dir
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>

4. Guidance
Use git

Do NOT use history re-editing (rebase)

Commit messages should be informative:

No: 'this should fix it', 'bump' commit messages

Yes: 'Resolve invalid API call in updating X'

Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.

Do NOT use the repository to upload data

Use VSCode or a similarly powerful IDE

Use Copilot for free

Sign up for GitHub Education

markdown
Copy code

---

### **What this includes**:
1. **Project Title and Summary**: The project is about detecting kidney stones from CT scans using deep learning models.
2. **Important Links**: Links for Timesheet, Slack channel, and Project report.
3. **Demo/Example Code**: Provides an example of how to use the Kidney-Net model.
4. **Folder Structure**: Describes where specific files are located, including data, source code, notebooks, and results.
5. **Installation**: Instructions for cloning the repo, setting up the environment, and installing dependencies.
6. **Reproduction**: Details on how to reproduce the results using the dataset.
7. **Guidance**: Best practices for using Git, coding conventions, and tool recommendations.

---

### **Next Steps**:
1. **Copy this content** into your **`README.md`** file.
2. **Commit and push** it to your repository:
   ```bash
   git add README.md
   git commit -m "Added detailed project README"
   git push origin main
