---
kernelspec:
  name: python
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
---

# Criteo_x1

```{note}
Please use the following evaluation settings for this benchmark:
+ Dataset split: [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1)
+ Rare features filtering: min_categr_count=1
+ Embedding size: 10
```

🔥 **See the benchmarking results**:

```{code-cell}
from plots import show_table, show_plot
show_plot("criteo_x1.csv")
show_table("criteo_x1.csv")
```
