import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

# === Génération du maillage fixe ===
def generate_fixed_mesh(num_points, x_range, y_range):
    np.random.seed(42)  # Reproductibilité
    nodes_x = np.random.uniform(*x_range, num_points)
    nodes_y = np.random.uniform(*y_range, num_points)
    points = np.vstack((nodes_x, nodes_y)).T

    # Création des arêtes via triangulation
    tri = Delaunay(points)
    edges = []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.append((simplex[i], simplex[j]))

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return nodes_x, nodes_y, edge_index

# === Génération des graphes avec champs de pression ===
def generate_pressure_graphs(num_graphs, nodes_x, nodes_y, edge_index, alpha_range, beta_range, params):
    graphs = []
    for _ in range(num_graphs):
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        x_c, y_c, a, b, P_0, epsilon, k = params

        pressures = alpha * P_0 * np.exp(-(((nodes_x - x_c)**2) / a**2 + ((nodes_y - y_c)**2) / b**2)) \
                    + beta * epsilon * np.sin(k * nodes_y)

        # Normaliser les pressions
        pressures = (pressures - pressures.min()) / (pressures.max() - pressures.min())
        x_coords = np.vstack((nodes_x, nodes_y)).T
        graph = Data(x=torch.tensor(x_coords, dtype=torch.float),
                     edge_index=edge_index,
                     y=torch.tensor(pressures, dtype=torch.float))
        graphs.append(graph)
    return graphs

# === Modèle GNN ===
class PressureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation_fn):
        super(PressureGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.activation_fn = activation_fn

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = self.activation_fn(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x

# === Entraînement du GNN ===
def train_gnn(model, train_loader, test_loader, optimizer, loss_fn, epochs):
    train_losses, test_r2_scores = [], []
    for epoch in tqdm(range(epochs), desc="Entraînement du modèle"):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(total_loss / len(train_loader))

        # Évaluation sur les données test
        test_r2 = evaluate_gnn(model, test_loader)
        test_r2_scores.append(test_r2)

    return train_losses, test_r2_scores

# === Évaluation du GNN ===
def evaluate_gnn(model, data_loader):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch).squeeze()
            all_preds.append(preds)
            all_targets.append(batch.y)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return r2_score(all_targets.numpy(), all_preds.numpy())

# === Tracé des courbes d'entraînement et de validation sur un même graphe ===
def plot_loss_and_r2(train_losses, test_r2_scores):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Tracé de la perte sur l'axe de gauche
    ax1.set_xlabel("Épochs")
    ax1.set_ylabel("Perte (MSE)", color="blue")
    ax1.plot(range(len(train_losses)), train_losses, label="Perte d'entraînement (MSE)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="upper left")

    # Tracé du R² sur l'axe de droite
    ax2 = ax1.twinx()  # Partager l'axe x
    ax2.set_ylabel("R² sur les données test", color="orange")
    ax2.plot(range(len(test_r2_scores)), test_r2_scores, label="R² sur les données test", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.legend(loc="upper right")

    # Titre et affichage
    plt.title("Perte d'entraînement et R² sur les données test")
    fig.tight_layout()
    plt.show()

def plot_single_test_case(graph, predictions, params, title="Test Case - Prédictions vs Réel"):
    y_max, y_min = params  # Dénormalisation des valeurs
    predictions_denorm = predictions * (y_max - y_min) + y_min
    targets_denorm = graph.y * (y_max - y_min) + y_min

    plt.figure(figsize=(12, 6))

    # Tracé des cibles réelles
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(
        graph.x[:, 0].numpy(),
        graph.x[:, 1].numpy(),
        c=targets_denorm.numpy(),
        cmap="viridis",
        s=10
    )
    plt.colorbar(scatter1, label="Pression (Réelle)")
    plt.title("Champ de pression réel (nœuds)")
    plt.axis("equal")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    # Tracé des prédictions
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(
        graph.x[:, 0].numpy(),
        graph.x[:, 1].numpy(),
        c=predictions_denorm.numpy(),
        cmap="viridis",
        s=10
    )
    plt.colorbar(scatter2, label="Pression (Prédite)")
    plt.title("Champ de pression prédit (nœuds)")
    plt.axis("equal")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    # Paramètres
    num_points = 5000
    x_range, y_range = (0, 10), (0, 10)
    params = (5, 5, 5, 3, 200, 10, 2 * np.pi / 10)  # Paramètres du champ de pression
    alpha_range, beta_range = (0.8, 1.2), (0.5, 1.5)

    # Génération du maillage
    nodes_x, nodes_y, edge_index = generate_fixed_mesh(num_points, x_range, y_range)

    # Génération des graphes
    num_train_graphs, num_test_graphs = 40, 10
    train_graphs = generate_pressure_graphs(num_train_graphs, nodes_x, nodes_y, edge_index, alpha_range, beta_range, params)
    test_graphs = generate_pressure_graphs(num_test_graphs, nodes_x, nodes_y, edge_index, alpha_range, beta_range, params)

    # Préparation des loaders
    train_loader = DataLoader(train_graphs, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=5, shuffle=False)

    # Initialisation du modèle
    num_layers = 4 # Nombre de convolution
    activation_fn = F.relu
    model = PressureGNN(in_channels=2, hidden_channels=16, out_channels=1, num_layers=num_layers, activation_fn=activation_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Entraînement
    epochs = 200
    train_losses, test_r2_scores = train_gnn(model, train_loader, test_loader, optimizer, loss_fn, epochs)

    # Évaluation finale
    test_r2 = evaluate_gnn(model, test_loader)
    print(f"R² sur les données test : {test_r2:.4f}")

    # Tracer les courbes
    plot_loss_and_r2(train_losses, test_r2_scores)

    # Visualisation d'un graphe de test
    test_graph = test_graphs[0]
    with torch.no_grad():
        predictions = model(test_graph)
    plot_single_test_case(test_graph, predictions.squeeze(), params=(200, 0))  # Dénormalisation avec P_max
