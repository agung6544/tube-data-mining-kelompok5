import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

def filter_rules_1to1(rules):
    return rules[
        (rules['antecedent_len'] == 1) &
        (rules['consequent_len'] == 1)
    ]

def calculate_tcr(rules_1to1):
    tcr = rules_1to1.groupby(
        rules_1to1['antecedents'].apply(lambda x: list(x)[0])
    )['confidence'].sum().reset_index()
    tcr.columns = ['product', 'TCR']
    tcr = tcr.sort_values(by='TCR', ascending=False)
    return tcr

def filter_arc(rules_1to1, conf_threshold=0.05, lift_threshold=0.8):
    arc = rules_1to1[
        (rules_1to1['confidence'] > conf_threshold) &
        (rules_1to1['lift'] > lift_threshold)
    ].copy()
    arc['antecedents'] = arc['antecedents'].apply(lambda x: list(x)[0])
    arc['consequents'] = arc['consequents'].apply(lambda x: list(x)[0])
    return arc[['antecedents', 'consequents', 'confidence', 'lift']]

def plot_tcr_bar(tcr, top_n=15):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tcr.head(top_n), x='TCR', y='product', palette='viridis')
    plt.title('Top 15 Produk dengan Total Contribution Ratio (TCR)')
    plt.xlabel('TCR')
    plt.ylabel('Produk')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_arc_graph(arc_chart):
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

def plot_product_layout(tcr, arc_chart):
    import numpy as np
    import matplotlib.pyplot as plt

    # Ambil top produk dari TCR
    produk_tcr = tcr.head(5)['product'].tolist()

    # Produk dari aturan asosiasi (ARC)
    produk_arc = list(set(arc_chart['antecedents']) | set(arc_chart['consequents']))

    # Gabungan semua produk
    semua_produk = list(set(produk_tcr) | set(produk_arc))

    # Ukuran grid
    grid_size = int(np.ceil(np.sqrt(len(semua_produk))))
    posisi_produk = {}
    grid_terisi = set()

    # Fungsi cari posisi sekitar pasangan
    def cari_posisi_terdekat(pos_awal):
        dxdy = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        for dx, dy in dxdy:
            calon = (pos_awal[0] + dx, pos_awal[1] + dy)
            if 0 <= calon[0] < grid_size and 0 <= calon[1] < grid_size and calon not in grid_terisi:
                return calon
        return None

    # 1. Tempatkan produk TCR di dekat pintu masuk
    pintu_masuk = (0, 0)
    antrian_pos = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
    for produk, pos in zip(produk_tcr, antrian_pos):
        posisi_produk[produk] = pos
        grid_terisi.add(pos)

    # 2. Tempatkan produk pasangan ARC di sekitar pasangannya
    for _, row in arc_chart.iterrows():
        ant = row['antecedents']
        cons = row['consequents']
        for produk, pasangan in [(ant, cons), (cons, ant)]:
            if produk not in posisi_produk and pasangan in posisi_produk:
                pos_pasangan = posisi_produk[pasangan]
                pos = cari_posisi_terdekat(pos_pasangan)
                if pos:
                    posisi_produk[produk] = pos
                    grid_terisi.add(pos)

    # 3. Tempatkan sisa produk
    for produk in semua_produk:
        if produk not in posisi_produk:
            for x in range(grid_size):
                for y in range(grid_size):
                    if (x, y) not in grid_terisi:
                        posisi_produk[produk] = (x, y)
                        grid_terisi.add((x, y))
                        break
                if produk in posisi_produk:
                    break

    # --- Visualisasi layout ---
    plt.figure(figsize=(12, 10))
    for produk, (x, y) in posisi_produk.items():
        warna = 'orange' if produk in produk_tcr else 'skyblue'
        plt.scatter(x, y, s=500, c=warna)
        plt.text(x, y, produk, ha='center', va='center', fontsize=9)

    # Pintu masuk dan panah
    plt.scatter(*pintu_masuk, s=800, c='green', label='Pintu Masuk')
    plt.text(pintu_masuk[0], pintu_masuk[1] - 0.5, "PINTU MASUK", ha='center', fontsize=10, fontweight='bold', color='green')
    plt.arrow(pintu_masuk[0], pintu_masuk[1], 2, 0, head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    plt.arrow(pintu_masuk[0] + 2, pintu_masuk[1], 0, 2, head_width=0.2, head_length=0.2, fc='gray', ec='gray')
    plt.text(pintu_masuk[0] + 1, pintu_masuk[1] + 0.2, '→', fontsize=14)
    plt.text(pintu_masuk[0] + 2.2, pintu_masuk[1] + 1, '↑', fontsize=14)

    plt.title("🛒 Tata Letak Produk: Produk Populer Dekat Pintu Masuk")
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.show()

