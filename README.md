# ComponentAnalysis
Machine Learning Componentry Analyzer

February 3rd, 2026

This code was built using the Xception, short for Extreme Inception, Deep Learning model developed at Google.

Using a suite of microscopic photos of sample clasts, this code will identify each clast in an image and then categorize them into a selected amount of groups. Once this process has been completed, the groups will be shared with the user who can then decide which groups are preferred, if any groups should be merged, or if any group needs to be further split.

# Instructions

## Setup (First Time Only)

### Prerequisites
- **Python 3.8+** installed
- **Node.js 16+** and npm installed

### Install Dependencies

1. **Install Python dependencies** (from the repo root):
   ```bash
   pip install -r requirements.txt
   ```

   > **Side note — use a virtual environment (if necessary):**
   >
   > If your system blocks global pip installs (PEP 668 / "externally managed"), create and use a venv:
   >
   > ```bash
   > python3 -m venv .venv
   > source .venv/bin/activate   # use `. .venv/bin/activate` on some shells
   > python -m pip install --upgrade pip setuptools wheel
   > python -m pip install -r requirements.txt
   > deactivate
   > ```
   >
   > Only use the venv when needed — the plain `pip install -r requirements.txt` is fine otherwise.

2. **Install Node.js dependencies** (from the repo root):
   ```bash
   cd electron
   npm install
   cd ..
   ```

## Start the app

From a terminal in the repo root (`ComponentAnalysis`):

```bash
cd electron
npm start
```

## Workflow

1. **Upload images (Step 1)**: Add your microscopy images.
2. **Extract grains (Step 2)**: Adjust padding/scale if needed, then run Extraction.
3. **Compute embeddings (Step 3)**: Run Embeddings.
4. **Cluster grains (Step 4)**: Choose the number of clusters and run Clustering.
5. **Review clusters (Clusters page)**:
   - Merge similar clusters.
   - Split mixed clusters.
   - Assign labels in the dropdowns (defaults to “to choose”).
6. **Set categories (Step 5)**:
   - Optionally set **Target categories** to cap the final classes.
   - Optionally set **Background label** and enable **Ignore background**.
7. **Train + predict (Clusters page)**: Train the model and optionally predict labels.
8. **Compute componentry**: Run Compute to generate class percentages.
9. **Export**: Use the Save buttons per page or Export from the header.

## Tips

- If you want fewer categories than clusters, set **Target categories** and map clusters into those labels.
- Background is excluded from results if its label matches the **Background label** and **Ignore background** is on.
