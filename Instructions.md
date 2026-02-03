# Instructions

## Start the app

From a terminal in the repo root (`ComponentAnalysis`):

```
cd C:\Python\ComponentryAnalysis\ComponentryAnalysis\ComponentAnalysis\electron
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
