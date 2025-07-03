import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro, levene
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu

df = pd.read_excel("distance_matrix.xlsx", index_col=0)
language_b_pages = [52, 62, 66, 68, 78, 80, 82, 86, 92, 96, 100, 110]

def get_language(filename):
    try:
        num = int(filename.split('_')[1])
        return "B" if num in language_b_pages else "A"
    except:
        return None

df.index = df.index.astype(str)
df.columns = df.columns.astype(str)

valid_pages = df.index.intersection(df.columns)

# Construct a dataframe with distances and group labels (A→A, B→B, A→B)
distance_data = []
for i in valid_pages:
    for j in valid_pages:
        if i == j:
            continue  
        dist = df.at[i, j]
        if pd.notna(dist):
            group_i = get_language(i)
            group_j = get_language(j)
            if group_i and group_j:
                if group_i == "A" and group_j == "A":
                    label = "A→A"
                elif group_i == "B" and group_j == "B":
                    label = "B→B"
                else:
                    label = "A→B"
                distance_data.append({"Pair": f"{i} vs {j}", "Distance": dist, "Group": label})


dist_df = pd.DataFrame(distance_data)

# Calculate and print the average distance for each group type
avg_distances = dist_df.groupby("Group")["Distance"].mean()
print("Average Distances:")
print(avg_distances)

# Create a boxplot to visualize distance distributions across groups
plt.figure(figsize=(8, 5))
sns.boxplot(x="Group", y="Distance", data=dist_df, hue="Group", palette={"A→A": "purple", "B→B": "red", "A→B": "blue"}, legend=False)
plt.title("Distribution of Distances by Group")
plt.xlabel("Group")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("distance_boxplot.png")
plt.show()

aa = dist_df[dist_df["Group"] == "A→A"]["Distance"]
bb = dist_df[dist_df["Group"] == "B→B"]["Distance"]
ab = dist_df[dist_df["Group"] == "A→B"]["Distance"]

# Perform Shapiro-Wilk tests to check for normality in each group
print("\nShapiro-Wilk Test for Normality:")
for group_name, group_data in zip(["A→A", "B→B", "A→B"], [aa, bb, ab]):
    stat, p = shapiro(group_data)
    result = "Normal" if p > 0.05 else "Not normal"
    print(f"{group_name}: p = {p:.4e} → {result}")

# Perform Levene’s test to check for homogeneity of variances across groups
print("\nLevene’s Test for Equal Variances:")
stat, p = levene(aa, bb, ab)
result = "Equal variances" if p > 0.05 else "Unequal variances"
print(f"p = {p:.4e} → {result}")

# Perform Kruskal-Wallis test (non-parametric ANOVA) to check overall group differences
h_stat, p_kw = kruskal(aa, bb, ab)
print("\nKruskal-Wallis Test:")
print(f"H-statistic: {h_stat:.4f}")
print(f"P-value:     {p_kw:.4e}")

pairs = [("A→A", aa, ab), ("A→A", aa, bb), ("A→B", ab, bb)]
labels = []
p_values = []

# Perform pairwise Wilcoxon Rank-Sum tests between groups
for label, g1, g2 in pairs:
    stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
    labels.append(f"{label} vs {label.replace('A', 'B')}")
    p_values.append(p_val)

# Apply Bonferroni correction to adjust for multiple comparisons
corrected = multipletests(p_values, method="bonferroni")
corrected_pvals = corrected[1]
reject_flags = corrected[0]

print("\nMann-Whitney U Tests (with Bonferroni correction):")
for i in range(len(labels)):
    result = " Significant" if reject_flags[i] else " Not significant"
    print(f"{labels[i]} → p = {p_values[i]:.4e}, corrected p = {corrected_pvals[i]:.4e} →{result}")