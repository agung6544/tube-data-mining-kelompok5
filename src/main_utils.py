# main_util.py

from model import generate_association_rules
from utils import (
    filter_rules_1to1,
    calculate_tcr,
    build_arc_chart,
    plot_tcr,
    plot_arc_graph,
)

# Asumsikan frequent_itemsets dan df_encoded sudah ada dari main.py
# Anda bisa import langsung jika dijadikan modul, atau simpan/load dari file

# Contoh skenario: file ini dijalankan setelah main.py menghasilkan frequent_itemsets
# Di sini kita lanjut dari frequent_itemsets → rules → util

def run_util_from_freq_itemsets(frequent_itemsets):
    rules = generate_association_rules(frequent_itemsets)

    # Filter aturan 1→1
    rules_1to1 = filter_rules_1to1(rules)
    print("\n📌 Aturan 1 item → 1 item:")
    print(rules_1to1[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    print(f"Jumlah aturan 1→1 ditemukan: {len(rules_1to1)}")

    # TCR
    tcr = calculate_tcr(rules_1to1)
    print("\n📊 Total Contribution Ratio (TCR):")
    print(tcr)

    # ARC
    arc_chart = build_arc_chart(rules_1to1)
    print("\n📐 Adjacency Relation Chart (ARC):")
    print(arc_chart)

    # Visualisasi TCR
    plot_tcr(tcr)

    # Visualisasi ARC Graph
    plot_arc_graph(arc_chart)

# Jalankan jika file ini dipanggil langsung
if __name__ == "__main__":
    from main import df_encoded, frequent_itemsets  # pastikan sudah diekspor dari main.py
    run_util_from_freq_itemsets(frequent_itemsets)
