My submission was built on the great work of [ragnar123](https://www.kaggle.com/ragnar123)'s LGBM Notebook, and [raddar](https://www.kaggle.com/raddar)'s cleaned integer dataset.

Challenges faced included fitting the full training set into memory and continuing to train as per usual (k-fold CV and then inference in the same notebook): this could not be done on Colab Free (10gb ram) or even Paperspace free tier (30gb ram). 

To resolve this, I used a specific seed and ran 5 Kaggle notebooks in parallel for 5-fold CV. The five resulting models were ensembled in a final Kaggle notebook.
