from data_loader import load_processed_data
from model import encode_transactions, generate_frequent_itemsets, generate_association_rules
from utils import filter_rules_1to1, calculate_tcr, filter_arc, plot_tcr_bar, plot_arc_graph, plot_product_layout

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

    # === TAMBAHAN VISUALISASI MULAI DARI SINI ===

    # Filter aturan 1→1
    rules_1to1 = filter_rules_1to1(rules)
    print("\n📌 Aturan 1 item → 1 item:")
    print(rules_1to1[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    print(f"Jumlah aturan 1→1 ditemukan: {len(rules_1to1)}")

    # Hitung TCR
    tcr = calculate_tcr(rules_1to1)
    print("\n📊 Total Contribution Ratio (TCR):")
    print(tcr)

    # Filter ARC
    arc_chart = filter_arc(rules_1to1)
    print("\n📐 Adjacency Relation Chart (ARC):")
    print(arc_chart)

    # Visualisasi
    plot_tcr_bar(tcr)
    plot_arc_graph(arc_chart)
    plot_product_layout(tcr, arc_chart)

if __name__ == "__main__":
    main()