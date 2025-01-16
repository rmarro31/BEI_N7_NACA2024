import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.cm as cm
import matplotlib.colors as colors
from tqdm import tqdm  # Barre de progression

# Données pour un graphe avec 10 nœuds (coordonnées fixes pour une grille simple)
x = torch.tensor([
    [0, 0], [1, 0], [2, 0], [0, 1], [1, 1],
    [2, 1], [0, 2], [1, 2], [2, 2], [1, 3]
], dtype=torch.float)

edge_index = torch.tensor(
    [
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 7],
        [1, 3, 0, 2, 1, 5, 0, 4, 3, 5, 2, 4, 3, 7, 4, 6, 5, 7, 7, 9]
    ],
    dtype=torch.long
)

y = torch.tensor([5.0, 7.0, 6.0, 8.0, 10.0, 9.0, 4.0, 11.0, 8.0, 12.0], dtype=torch.float)

# Normalisation des données
x_max, x_min = x.max(dim=0).values, x.min(dim=0).values
x_normalized = (x - x_min) / (x_max - x_min)

y_max, y_min = y.max(), y.min()
y_normalized = (y - y_min) / (y_max - y_min)

# Création du graphe
graph = Data(x=x_normalized, edge_index=edge_index, y=y_normalized)

# Modèle GNN avec sigmoïde comme fonction d'activation
class PressureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PressureGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.sigmoid(self.conv1(x, edge_index))  # Première couche
        x = torch.sigmoid(self.conv2(x, edge_index))  # Deuxième couche
        x = torch.sigmoid(self.conv3(x, edge_index))  # Troisième couche
        x = torch.sigmoid(self.conv4(x, edge_index))  # Quatrième couche
        return x

# Initialisation du modèle et des outils d'entraînement
model = PressureGNN(in_channels=2, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Entraînement du modèle
losses = []
epochs = 1000  # Ajuster selon les besoins

for epoch in tqdm(range(epochs), desc="Entraînement du modèle"):
    model.train()
    optimizer.zero_grad()
    out = model(graph)
    loss = loss_fn(out.squeeze(), graph.y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

# Tracer la fonction de perte
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), losses, label="Perte d'entraînement", color="blue")
plt.xlabel("Époque")
plt.ylabel("Perte (MSE)")
plt.title("Évolution de la perte pendant l'entraînement (Sigmoïde)")
plt.legend()
plt.grid(True)
plt.show()

# Prédictions du modèle
model.eval()
with torch.no_grad():
    predictions = model(graph)
print(f"Prédictions : {predictions.squeeze().tolist()}")

# Calculer les métriques
mse = mean_squared_error(graph.y, predictions.squeeze())
r2 = r2_score(graph.y, predictions.squeeze())
print(f"MSE: {mse:.4f}, R2 Score: {r2:.4f}")

# Visualisation des prédictions avec les valeurs réelles
def plot_predictions_with_targets(graph, predictions, targets, title="Prédictions vs Valeurs Réelles"):
    G = nx.Graph()
    for i, coords in enumerate(graph.x):
        G.add_node(i, pos=(coords[0].item(), coords[1].item()), pressure=predictions[i].item())
    edges = graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, 'pos')
    pressures = nx.get_node_attributes(G, 'pressure')

    norm = colors.Normalize(vmin=min(pressures.values()), vmax=max(pressures.values()))
    cmap = cm.viridis

    node_colors = [pressures[node] for node in G.nodes()]
    fig, ax = plt.subplots(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap=cmap, node_size=500, alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(G, pos, edge_color="black", alpha=0.5, ax=ax)
    labels = {node: f"P: {pressures[node]:.2f}\nT: {targets[node]:.2f}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black", ax=ax)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(node_colors)
    cbar = plt.colorbar(sm, ax=ax, label="Pression")
    plt.title(title)
    plt.show()

plot_predictions_with_targets(graph, predictions.squeeze(), graph.y)
