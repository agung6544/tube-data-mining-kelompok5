# util.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def filter_rules_1to1(rules):
    """
    Filter aturan asosiasi 1 → 1 dari hasil generate_association_rules()
    """
    rules_1to1 = rules[
        (rules['antecedent_len'] == 1) & 
        (rules['consequent_len'] == 1)
    ]
    return rules_1to1

def calculate_tcr(rules_1to1):
    """
    Menghitung Total Contribution Ratio (TCR) untuk setiap produk (antecedent)
    """
    tcr = rules_1to1.groupby(rules_1to1['antecedents'].apply(lambda x: list(x)[0]))['confidence'].sum().reset_index()
    tcr.columns = ['product', 'TCR']
    return tcr.sort_values(by='TCR', ascending=False)

def build_arc_chart(rules_1to1, min_confidence=0.05, min_lift=0.8):
    """
    Filter aturan untuk ARC dan siapkan format adjacency chart
    """
    arc = rules_1to1[
        (rules_1to1['confidence'] > min_confidence) & 
        (rules_1to1['lift'] > min_lift)
    ]
    arc_chart = arc[['antecedents', 'consequents', 'confidence', 'lift']].copy()
    arc_chart['antecedents'] = arc_chart['antecedents'].apply(lambda x: list(x)[0])
    arc_chart['consequents'] = arc_chart['consequents'].apply(lambda x: list(x)[0])
    return arc_chart

def plot_tcr(tcr, top_n=15):
    """
    Visualisasi barplot TCR produk teratas
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tcr.head(top_n), x='TCR', y='product', palette='viridis')
    plt.title('Top Produk dengan Total Contribution Ratio (TCR)')
    plt.xlabel('TCR')
    plt.ylabel('Produk')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_arc_graph(arc_chart):
    """
    Visualisasi directed graph untuk ARC (adjacency relation chart)
    """
    G = nx.DiGraph()

    for _, row in arc_chart.iterrows():
        G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'], label=f"{row['confidence']:.2f}")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("ARC: Produk yang Cocok Diletakkan Berdekatan")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_store_layout(posisi_produk, pintu_masuk=(0, 0)):
    """
    Visualisasi peta tata letak produk + pintu masuk + arah panah alur
    """
    plt.figure(figsize=(10, 10))

    # Plot produk
    for produk, (x, y) in posisi_produk.items():
        plt.scatter(x, y, s=500, c='skyblue')
        plt.text(x, y, produk, ha='center', va='center', fontsize=9)

    # Tambahkan pintu masuk
    plt.scatter(*pintu_masuk, s=800, c='green', label='Pintu Masuk')
    plt.text(pintu_masuk[0], pintu_masuk[1] - 0.5, "PINTU MASUK", ha='center', fontsize=10, fontweight='bold', color='green')

    # Arah panah simulasi alur
    plt.arrow(pintu_masuk[0], pintu_masuk[1], 2, 0, head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    plt.arrow(pintu_masuk[0] + 2, pintu_masuk[1], 0, 2, head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    plt.text(pintu_masuk[0] + 1, pintu_masuk[1] + 0.2, '→', fontsize=14)
    plt.text(pintu_masuk[0] + 2.2, pintu_masuk[1] + 1, '↑', fontsize=14)

    plt.title("🛒 Peta Tata Letak Produk (dengan Arah Pintu Masuk)")
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.show()
