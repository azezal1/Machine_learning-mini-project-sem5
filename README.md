# Mall Customer Segmentation using K-Means

ML Mini Project, Semester 5, Unit 3 (Mixture Models and EM).

K-Means clustering groups 200 mall customers by **annual income** and **spending score** so the mall can target each group with different marketing.

## Files

| File | What it is |
|---|---|
| `index.html` | Interactive website: live clusters, step-by-step K-Means animation, elbow and silhouette charts, segment profiles, new-customer predictor |
| `kmeans_customer_segmentation.py` | Python code (pandas, scikit-learn, matplotlib) |
| `Mall_Customers.csv` | Dataset, 200 customers |

## Run the website

Open `index.html` in any browser. No install needed.

To host it online: repo **Settings → Pages → Deploy from branch → main / (root)**. It will be live at `https://azezal1.github.io/Machine_learning-mini-project-sem5/`.

## Run the Python code

```bash
pip install pandas scikit-learn matplotlib
python kmeans_customer_segmentation.py
```

Or upload the `.py` and `.csv` to Google Colab.

## Method

1. Scale income and spending score with `StandardScaler`.
2. Choose K with the **elbow method** (WCSS) and **silhouette score**. Both point to **K = 5**.
3. Train K-Means (k-means++ init, 10 restarts).
4. Interpret the 5 segments:

| Segment | Profile | Marketing idea |
|---|---|---|
| Target | High income, high spending | VIP and loyalty programs |
| Careful | High income, low spending | Quality-focused promotions |
| Standard | Average income and spending | Regular seasonal offers |
| Careless | Low income, high spending | Trendy, low-cost products |
| Sensible | Low income, low spending | Discounts and value packs |

## Link to EM

K-Means is a hard-assignment special case of EM for Gaussian mixtures: the assignment step is the E-step and the centroid update is the M-step.
