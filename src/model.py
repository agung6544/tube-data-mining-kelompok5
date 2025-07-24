# model.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def encode_transactions(transactions):
    """
    One-hot encode transaksi menjadi DataFrame boolean.
    """
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df_encoded

def generate_frequent_itemsets(df_encoded, min_support=0.002):
    """
    Menghasilkan frequent itemsets menggunakan algoritma Apriori.
    """
    return apriori(df_encoded, min_support=min_support, use_colnames=True)

def generate_association_rules(frequent_itemsets, min_confidence=0.05):
    """
    Menghasilkan aturan asosiasi dari itemsets yang sering muncul.
    """
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    return rules
