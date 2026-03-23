# Stone-See (KidneyNet) — Technical Description

## Purpose and domain

Stone-See (branded in code as **KidneyNet**) is a **binary image classifier** for **kidney stone detection** from CT-style scan images. It combines **PyTorch deep learning**, a **Flask inference API**, and a **Next.js** single-page upload UI. The project is aligned with SFU CMPT 340 coursework (dataset on Google Drive, team workflow in README).

Use this document to extract bullets for roles in **ML/CV**, **full-stack**, **health tech (research tooling)**, or **DevOps/scripting**.

---

## Repository layout (logical)

| Area | Role |
|------|------|
| `src/preprocessing/load_data.py` | Dataset loading, train/val/test splits, augmentations |
| `src/models/train_model.py` | ResNet-18 training loop, checkpoints, class weights |
| `src/evaluation/` | Offline evaluation, comparison, reports, single-image CLI predict |
| `backend/app.py` | Production-style REST API for uploaded images |
| `frontend/` | Next.js UI calling the API |
| `run.py` | Cross-platform orchestration (Flask + Next + browser) |
| `results/` | Saved weights (`resnet18_kidney_best.pth`), metrics JSON, figures |

---

## Machine learning stack

### Model architecture

- **Backbone:** `torchvision.models.resnet18` with **`weights=None`** (training from scratch in the provided script, not ImageNet-pretrained in code).
- **Head:** Final fully connected layer replaced with `nn.Linear(in_features, 2)` for **two classes** (indices correspond to `ImageFolder` class order: typically `Non-Stone` vs `Stone` as used in evaluation strings).
- **Input size:** **224×224** RGB, consistent across training, evaluation, Flask preprocess, and CLI predict.

### Training data pipeline (`load_data.py`)

Two entry patterns:

1. **`load_kidney_data(data_dir)`**  
   - Uses `torchvision.datasets.ImageFolder` on a single root folder with class subdirectories.  
   - **Train transform:** resize 224, random horizontal flip (p=0.5), random rotation ±10°, ToTensor, ImageNet normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).  
   - **Val/test transform:** resize 224, ToTensor, same normalization (no random geom).  
   - **Split:** `random_split` on indices into **train / val / test** with `val_split=0.2` meaning val and test each get `floor(n * 0.2)` and train gets the remainder (see code for exact sizing).

2. **`load_combined_kidney_data(augmented_dir, original_dir)`**  
   - Loads **two** `ImageFolder` datasets (no transform on load), concatenates with `ConcatDataset`.  
   - Custom **`TransformDataset`** wrapper resolves `ConcatDataset` indexing (which sub-dataset and local index) and applies train vs val transforms per split subset.  
   - Same **train/val/test** random split logic on the **combined** index range.

**Training script** (`train_model.py`) prefers **both** `data/Augmented_Dataset` and `data/Original_Dataset` when present; otherwise falls back to augmented-only, original-only, or generic `data/`.

### Loss, optimization, and regularization

- **Loss:** `CrossEntropyLoss` with **class weights** `tensor([1.0, 1.5])` on device — **Stone** weighted higher to mitigate imbalance / false negatives.
- **Optimizer:** **Adam**, learning rate **`5e-5`**.
- **Scheduler:** `ReduceLROnPlateau` on **validation loss**, factor `0.5`, patience `3`.
- **Epochs:** **10** in `main()`.

### Checkpoint format

- **Best model** (highest val accuracy): `results/models/resnet18_kidney_best.pth` as a **dict** with keys including `model_state_dict`, `optimizer_state_dict`, `val_accuracy`, `val_loss`, `epoch`.
- **Final model:** `results/models/resnet18_kidney.pth` includes `history` in the dict.
- **History JSON:** `results/models/resnet18_kidney_history.json` — per-epoch train/val loss and accuracy, learning rates.

### Device selection

Consistent pattern: **Apple MPS** if available, else **CUDA**, else **CPU** — used in training, evaluation, predict CLI, and Flask load.

---

## Evaluation and reporting

### `evaluate_model.py`

- Loads best (or final) checkpoint, runs **test** `DataLoader` with `model.eval()` and `torch.no_grad()`.
- Collects predictions, labels, and **softmax probabilities**.
- **Metrics:** `sklearn` accuracy, weighted precision/recall/F1, per-class metrics, confusion matrix, **ROC curve** using probability of positive class (Stone), **AUC**.
- **Plots:** seaborn heatmap confusion matrix, ROC curve → `results/figures/`.
- **Exports:** `results/evaluation_metrics.json` (JSON-serializable subset).

### `compare_datasets.py`

- Evaluates the **same** trained model on **different dataset splits** (e.g. augmented vs original) to compare generalization; uses similar metric computation and plotting (see file for full graph set).

### `generate_report.py`

- Reads `evaluation_metrics.json` and optional training history JSON.
- Builds structured **report sections** (title, date, overall metrics, per-class, narrative fields) for **text/LaTeX-style output** (functions for formatting numbers as percentages, etc.).

### `predict_image.py`

- CLI: `python src/evaluation/predict_image.py <path>` — loads model, preprocesses file, prints argmax class and confidence from softmax.

---

## Backend API (`backend/app.py`)

**Framework:** Flask + **flask_cors** (open CORS for browser clients).

**Endpoints:**

- **`GET /health`** — JSON `status`, `model_loaded`.
- **`GET /metrics`** — Serves `results/evaluation_metrics.json` if present.
- **`POST /predict`** — Multipart form field **`image`** (file).  
  - Preprocess: same 224 resize, ToTensor, ImageNet normalize as training val path.  
  - Forward pass, **softmax** over 2 logits.  
  - **Decision rule (important for interviews):** Not plain argmax — uses **`stone_threshold = 10.0`** on **Stone** class probability (%): if `stone_prob >= 10`, predict **Stone** and report confidence as stone probability; else **Non-Stone** with confidence as non-stone probability. This **increases sensitivity** to stones (fewer missed stones at the cost of more false positives).

**Model load:** Prefers `resnet18_kidney_best.pth`, falls back to `resnet18_kidney.pth`; supports both dict checkpoints and raw `state_dict`.

**Server:** `app.run(host='0.0.0.0', port=5000, debug=True)`.

---

## Frontend (`frontend/`)

- **Next.js** app (see `package.json` / `next.config.js`). Main UI: **`pages/index.js`** (not App Router).
- **State:** React `useState` for file, preview data URL, loading, API result, error.
- **API base:** `process.env.NEXT_PUBLIC_API_URL` or default `http://localhost:5000`.
- **Flow:** User selects image → `FileReader` preview → `FormData` POST to `/predict` → renders prediction, confidence, and **bar visualization** for both class probabilities.
- **Styling:** Large inline `styled-jsx` block — gradient hero, cards, responsive tweaks.
- **SEO:** `next/head` title and meta description; disclaimer footer (research use only).

**Deployment note:** `vercel.json` may exist for static/SSR hosting; backend must be reachable at configured `NEXT_PUBLIC_API_URL`.

---

## Orchestration (`run.py`)

Python 3.7+ **orchestrator** (KidneyNet branding in banner):

1. Checks Python version, presence of `backend/app.py` and `frontend/`, optional `npm`.
2. **Kills** prior servers: Windows uses `netstat`/`taskkill` patterns; Unix uses `pkill` for `python.*app.py` and `next dev`.
3. Starts **Flask** as subprocess with cwd `backend/`, logs to `logs/backend.log`.
4. Polls `http://localhost:5000/health`.
5. Runs `npm install` in `frontend/` if `node_modules` missing; starts **`npm run dev`**, logs to `logs/frontend.log`.
6. Opens browser to `http://localhost:3000`.
7. **Signal handlers** SIGINT/SIGTERM → terminate child processes and repeat port cleanup.

**Interview angles:** process management, cross-platform subprocess/logging, graceful shutdown.

---

## Dependencies (conceptual)

- **Python:** `torch`, `torchvision`, `flask`, `flask_cors`, `PIL`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `tqdm`, etc. (exact pins may live in `requirements.txt` / conda yaml in repo).
- **Node:** Next.js stack per `frontend/package.json`.

---

## How to summarize by role

- **ML engineer:** ResNet-18 binary classifier; weighted CE; combined augmented+original data; full sklearn + ROC; sensitive threshold at inference.
- **Backend:** Flask REST, file upload, PyTorch load, MPS/CUDA/CPU.
- **Frontend:** Next.js + fetch + FormData; env-based API URL.
- **MLOps light:** `run.py` local stack, checkpoint/history JSON, evaluation artifacts.

---

## Caveats for accurate talking points

- Medical **disclaimer** in UI: research / not diagnostic.
- Inference **threshold** in API differs from strict **argmax** used in offline `predict_image.py` — be explicit which path you discuss.
- Dataset **not** in git per course rules; paths expect `data/Augmented_Dataset`, `data/Original_Dataset`, or `data/` with class folders.
