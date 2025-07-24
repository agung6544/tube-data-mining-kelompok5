# main.py

from data_loader import load_processed_data
from model import encode_transactions, generate_frequent_itemsets, generate_association_rules

def main():
    # Load data dari folder data/processed/
    df_processed = load_processed_data('transactions.csv')

    if df_processed is None:
        return

    # Pisahkan item dalam transaksi berdasarkan koma
    transactions = df_processed['Transaction'].apply(lambda x: x.split(','))

    # One-hot encoding
    df_encoded = encode_transactions(transactions)

    # Apriori untuk frequent itemsets
    frequent_itemsets = generate_frequent_itemsets(df_encoded, min_support=0.002)
    print("✅ Frequent Itemsets:")
    print(frequent_itemsets)

    # Generate association rules
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.05)

    # Tampilkan aturan asosiasi (top 10)
    print("\n🔎 Aturan Asosiasi (tanpa filter panjang):")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
    print(f"📌 Jumlah total aturan: {len(rules)}")

if __name__ == "__main__":
    main()
