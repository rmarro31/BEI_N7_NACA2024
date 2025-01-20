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
], dtype=torch.float)  # Coordonnées des nœuds (on pourra normalement automatiser avec le fichier du maillage)

edge_index = torch.tensor(
    [
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 7],
        [1, 3, 0, 2, 1, 5, 0, 4, 3, 5, 2, 4, 3, 7, 4, 6, 5, 7, 7, 9]
    ],
    dtype=torch.long
)  # Connexions fixes pour une grille (à voir si on peut automatiser aussi )

y = torch.tensor([5.0, 7.0, 6.0, 8.0, 10.0, 9.0, 4.0, 11.0, 8.0, 12.0], dtype=torch.float)  # Valeurs cibles (valeurs pour chaque noeuds)
#Données à prendre dans le dossier de sortie de Fluent.


# Création du graphe
graph = Data(x=x, edge_index=edge_index, y=y) #Rassemble toutes les infos pour entrainer le model. 

# Visualisation du graphe
def plot_graph(graph, title="Graphe du maillage"):
    G = nx.Graph()
    for i, coords in enumerate(graph.x):
        G.add_node(i, pos=(coords[0].item(), coords[1].item()))
    edges = graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=500, edge_color="black")
    plt.title(title)
    plt.show()

plot_graph(graph)

# Modèle GNN avec 4 couches de convolution
class PressureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PressureGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):  # Cette fonction est appelé nul par car elle est automatiquement appelé lorqu'on appel le model
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Première couche
        x = F.relu(self.conv2(x, edge_index))  # Deuxième couche
        x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv4(x, edge_index)) # Troisième couche
        x = self.conv5(x, edge_index)          # Quatrième couche
        return x

# Initialisation du modèle et des outils d'entraînement
model = PressureGNN(in_channels=2, hidden_channels=16, out_channels=1)  # créer une instance de la classe PressureGNN
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #L'optimiseur Adam ajuste dynamiquement 
#le taux d'apprentissage pour chaque paramètre
#en fonction des moyennes mobiles des gradients et de leurs carrés.
loss_fn = nn.MSELoss()

# Entraînement du modèle
losses = []
epochs = 500


for epoch in tqdm(range(epochs), desc="Entraînement du modèle"):
    model.train()  # Cela permet d'indiquer que l'on va ajuster les poids du modèle pendant cette période.
    optimizer.zero_grad()  # Avant chaque propagation arrière (backpropagation),
    #les gradients accumulés doivent être réinitialisés à 0. 
    #Cela permet d'éviter que les gradients des itérations précédentes
    #ne s'ajoutent à ceux de l'itération actuelle.



    out = model(graph)  # Passe les données d'entrée (graph) à travers le modèle.
    #Le modèle effectue des calculs sur les nœuds 
    #et les arêtes du graphe pour générer les prédictions.
    loss = loss_fn(out.squeeze(), graph.y) #Cette ligne calcule la perte (erreur)
    #entre les prédictions (out) et les valeurs réelles (graph.y).
    losses.append(loss.item())
    loss.backward()
    "Effectue la propagation arrière (backpropagation)"
    "pour calculer les gradients de la perte par rapport aux paramètres du modèle (poids des couches)." 
    "Ces gradients indiquent dans quelle direction chaque poids doit être ajusté pour réduire l'erreur."
    optimizer.step() # met à jour les poids de chaque convolution (avec un algo de descente de gradiant)

# Tracer la fonction de perte
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), losses, label="Perte d'entraînement", color="blue")
plt.xlabel("Époque")
plt.ylabel("Perte (MSE)")
plt.title("Évolution de la perte pendant l'entraînement")
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
    # Créer le graphe NetworkX avec les prédictions et les cibles
    G = nx.Graph()
    for i, coords in enumerate(graph.x):
        G.add_node(i, pos=(coords[0].item(), coords[1].item()), pressure=predictions[i].item())
    edges = graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, 'pos')
    pressures = nx.get_node_attributes(G, 'pressure')

    # Normaliser les pressions pour les couleurs
    norm = colors.Normalize(vmin=min(pressures.values()), vmax=max(pressures.values()))
    cmap = cm.viridis

    # Appliquer les couleurs aux nœuds
    node_colors = [pressures[node] for node in G.nodes()]
    
    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Dessiner les nœuds et les arêtes
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap=cmap, node_size=500, alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(G, pos, edge_color="black", alpha=0.5, ax=ax)
    
    # Affichage des valeurs de pression (prédictions et cibles)
    labels = {node: f"P: {pressures[node]:.2f}\nT: {targets[node]:.2f}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black", ax=ax)

    # Ajouter la barre de couleur
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(node_colors)
    cbar = plt.colorbar(sm, ax=ax, label="Pression")

    # Titre et affichage
    plt.title(title)
    plt.show()

plot_predictions_with_targets(graph, predictions.squeeze(), graph.y)
