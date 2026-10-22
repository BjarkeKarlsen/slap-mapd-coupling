"""
Shared setup for the Problem Formulation chapter figures: the warehouse
graph, its layout, the shared colour palette, and the common draw helpers
every fig_*.py module builds on.

OUT is what both problem_formulation*.tex point at via
\\graphicspath{{\\subfix{../Images/}}} -- keep it in sync with that, not the
other way around.
"""
import os
import matplotlib
matplotlib.use("Agg")
import networkx as nx

OUT = "thesis-progress/scripts/Images"
os.makedirs(OUT, exist_ok=True)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",          # match a LaTeX thesis
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

#TODO fix the colors to match the thesis color palette, here rcv_f and rcv_e are not the same as in the thesis, they are too light and not visible enough
C = dict(edge="#9aa5b1", node="#5a6673", store_f="#cfe3f7", store_e="#2f6fae",
         del_f="#ffd9b3", del_e="#c9761a", rcv_f="#f8c8a0", rcv_e="#d48c4a", ep_f="#d7f0d8", ep_e="#3f8f45",
         agent="#c0392b", accent="#2f6fae", comm="#7d3c98", text="#43505c")


# ---------------------------------------------------------------- layout
# G is directed: (v,w) in E does not imply (w,v) in E (sec:pf:env). Every
# adjacency is added in both directions except ONE_WAY, which models a
# genuine one-way aisle segment -- not a drawing convention, an actual
# asymmetry in the edge set, so a figure caption can honestly say so.
ONE_WAY = {("1_8", "2_8")}  # allowed direction only: 1_8 -> 2_8


def warehouse_graph(rows=5, cols=9, dx=1.35, dy=1.15):
    """Aisle layout: two cross-aisles (top/bottom row) + vertical racks."""
    G, pos = nx.DiGraph(), {}
    for r in range(rows):
        for c in range(cols):
            if r in (0, rows - 1) or c % 2 == 0:
                n = f"{r}_{c}"
                G.add_node(n)
                pos[n] = (c * dx, -r * dy)
    for n in list(G):
        r, c = map(int, n.split("_"))
        for dr, dc in ((0, 1), (1, 0)):
            m = f"{r+dr}_{c+dc}"
            if m not in G:
                continue
            if (n, m) in ONE_WAY or (m, n) in ONE_WAY:
                u, v = next(iter(ONE_WAY & {(n, m), (m, n)}))
                G.add_edge(u, v)
            else:
                G.add_edge(n, m)
                G.add_edge(m, n)
    return G, pos


G, POS = warehouse_graph()
STORAGE = [n for n in G if 0 < int(n.split("_")[0]) < 4 and int(n.split("_")[1]) % 2 == 0]
DELIVERY = ["0_0", "4_8"]
ENDPOINTS = ["0_4", "4_2", "0_8"]


RECEIVE = ["0_2", "4_6"]


def draw_base(ax, dim=False):
    """Draw the warehouse graph; dim=True greys it out as a background."""
    a = 0.35 if dim else 1.0
    two_way = [(u, v) for u, v in G.edges() if (v, u) in G.edges()]
    #one_way = [(u, v) for u, v in G.edges() if (v, u) not in G.edges()]
    nx.draw_networkx_edges(G, POS, edgelist=two_way, ax=ax, edge_color=C["edge"],
                           width=1.6, alpha=a, arrows=False)
    #nx.draw_networkx_edges(G, POS, edgelist=one_way, ax=ax, edge_color=C["accent"],
    #                       width=2.0, alpha=a, arrows=True, arrowsize=18,
    #                       node_size=460, connectionstyle="arc3,rad=0.0")
    other = [n for n in G if n not in STORAGE + DELIVERY + RECEIVE + ENDPOINTS]
    for nl, fc, ec, shape, size in (
        (other, "white", C["node"], "o", 460),
        (STORAGE, C["store_f"], C["store_e"], "o", 460),
        (DELIVERY, C["del_f"], C["del_e"], "s", 560),
        (ENDPOINTS, C["ep_f"], C["ep_e"], "D", 470),
        (RECEIVE, C["rcv_f"], C["rcv_e"], "v", 560),
    ):
        nx.draw_networkx_nodes(G, POS, nodelist=nl, node_color=fc, edgecolors=ec,
                               node_shape=shape, node_size=size, ax=ax, alpha=a)

def draw_base_mini(ax, dim=False):
    """Draw the warehouse graph; dim=True greys it out as a background."""
    a = 0.35 if dim else 1.0
    two_way = [(u, v) for u, v in G.edges() if (v, u) in G.edges()]
    #one_way = [(u, v) for u, v in G.edges() if (v, u) not in G.edges()]
    nx.draw_networkx_edges(G, POS, edgelist=two_way, ax=ax, edge_color=C["edge"],
                           width=1.6, alpha=a, arrows=False)
    #nx.draw_networkx_edges(G, POS, edgelist=one_way, ax=ax, edge_color=C["accent"],
    #                       width=2.0, alpha=a, arrows=True, arrowsize=18,
    #                       node_size=460, connectionstyle="arc3,rad=0.0")
    other = [n for n in G if n not in STORAGE + DELIVERY + RECEIVE + ENDPOINTS]
    for nl, fc, ec, shape, size in (
        (other, "white", C["node"], "o", 460),
        (STORAGE, C["store_f"], C["store_e"], "o", 460),
        (DELIVERY, C["del_f"], C["del_e"], "s", 560),
        (ENDPOINTS, C["ep_f"], C["ep_e"], "D", 470),
        (RECEIVE, C["rcv_f"], C["rcv_e"], "v", 560),
    ):
        nx.draw_networkx_nodes(G, POS, nodelist=nl, node_color=fc, edgecolors=ec,
                               node_shape=shape, node_size=size, ax=ax, alpha=a)




def put_agent(ax, node, label, dx=0, dy=16):
    ax.scatter(*POS[node], s=260, c=C["agent"], zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(label, POS[node], textcoords="offset points", xytext=(dx, dy),
                ha="center", color=C["agent"], fontsize=14)
