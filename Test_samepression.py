## Importé le 06/01/2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
from scipy.spatial import Delaunay
from sklearn.metrics import r2_score




# --- Champ de pression ---
x_c, y_c = 5, 5  # Centre de l'aile dans le plan (10x10)
a, b = 5, 3      # Demi-axes de l'ellipse
P_0 = 200        # Pression maximale
epsilon = 10     # Amplitude de la perturbation
k = 2 * np.pi / 10  # Fréquence de la perturbation

x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
xv, yv = np.meshgrid(x, y)
pressure_field = P_0 * np.exp(-(((xv - x_c) ** 2) / a ** 2 + ((yv - y_c) ** 2) / b ** 2)) \
                 + epsilon * np.sin(k * yv)

def calculate_pressure(x, y):
    return P_0 * np.exp(-(((x - x_c) ** 2) / a ** 2 + ((y - y_c) ** 2) / b ** 2)) + epsilon * np.sin(k * y)

# --- Normalisation ---
def normalize_data(x, y):
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    y_min, y_max = y.min(), y.max()
    x_normalized = (x - x_min) / (x_max - x_min)
    y_normalized = (y - y_min) / (y_max - y_min)
    return x_normalized, y_normalized, x_min, x_max, y_min, y_max

def denormalize_data(y_normalized, y_min, y_max):
    return y_normalized * (y_max - y_min) + y_min

# --- Génération des graphes ---
def generate_graph(num_points, x_min=0, x_max=10, y_min=0, y_max=10):
    G = nx.Graph()
    nodes_x = np.random.uniform(x_min, x_max, num_points)
    nodes_y = np.random.uniform(y_min, y_max, num_points)
    pressures = [calculate_pressure(nodes_x[i], nodes_y[i]) for i in range(num_points)]
    # Normaliser les données
    x_coords = np.vstack((nodes_x, nodes_y)).T
    x_normalized, pressures_normalized, x_min_vals, x_max_vals, y_min_val, y_max_val = normalize_data(x_coords, np.array(pressures))
    # Création des nœuds
    for i in range(num_points):
        G.add_node(i, pos=(x_normalized[i][0], x_normalized[i][1]), pressure=pressures_normalized[i])
    # Création des arêtes via triangulation
    tri = Delaunay(x_coords)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])
    # Retourner les données normalisées et les paramètres de normalisation
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x_tensor = torch.tensor(x_normalized, dtype=torch.float)
    y_tensor = torch.tensor(pressures_normalized, dtype=torch.float)
    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor), (x_min_vals, x_max_vals, y_min_val, y_max_val)

def generate_graphs(num_train_graphs, num_test_graphs, num_points_train, num_points_test):
    train_graphs, test_graphs = [], []
    train_params, test_params = [], []
    for _ in range(num_train_graphs):
        graph, params = generate_graph(num_points_train)
        train_graphs.append(graph)
        train_params.append(params)
    for _ in range(num_test_graphs):
        graph, params = generate_graph(num_points_test)
        test_graphs.append(graph)
        test_params.append(params)
    return train_graphs, test_graphs, train_params, test_params

# --- Modèle GNN ---
class PressureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation_fn):
        super(PressureGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation_fn = activation_fn
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))  # Dernière couche sans activation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Passer par les couches intermédiaires avec ReLU
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x, edge_index))
        # Dernière couche avec Sigmoid
        x = self.layers[-1](x, edge_index)
        x = torch.sigmoid(x)  # Activation sigmoid pour la dernière couche
        return x


# --- Entraînement avec suivi des performances ---
def train_gnn_with_validation(model, train_loader, test_loader, train_params, test_params, optimizer, loss_fn, epochs):
    model.train()
    train_losses = []
    test_r2_scores = []

    for epoch in tqdm(range(epochs), desc="Entraînement"):
        # Entraînement
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation sur les données de test
        _, test_r2 = evaluate_gnn(model, test_loader, test_params)
        test_r2_scores.append(test_r2)

    return train_losses, test_r2_scores

# --- Évaluation ---
def evaluate_gnn(model, loader, params_list):
    model.eval()
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for batch, params in zip(loader, params_list):
            out = model(batch)
            y_pred_denormalized = denormalize_data(out.squeeze().numpy(), params[2], params[3])
            y_target_denormalized = denormalize_data(batch.y.numpy(), params[2], params[3])
            all_targets.extend(y_target_denormalized)
            all_predictions.extend(y_pred_denormalized)
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    return mse, r2

# --- Tracer les courbes ---
def plot_training_and_validation(train_losses, test_r2_scores):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Courbe de perte d'entraînement
    color = 'tab:blue'
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte (MSE)", color=color)
    ax1.plot(train_losses, label="Perte d'entraînement", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")

    # Courbe de R² sur les données de test
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel("R² (Données de test)", color=color)
    ax2.plot(test_r2_scores, label="R² (Test)", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.title("Perte d'entraînement et R² (Test)")
    plt.grid(True)
    plt.show()


def plot_pressure_field(true_pressures, predicted_pressures, x_coords, y_coords):
    # Conversion des pressions de test et prédites en numpy pour l'affichage
    true_pressures = np.array(true_pressures)
    predicted_pressures = np.array(predicted_pressures)
    
    # Création de la figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Visualisation des pressions réelles
    sc = ax[0].scatter(x_coords, y_coords, c=true_pressures, cmap='viridis', s=10)
    ax[0].set_title("Pression réelle")
    fig.colorbar(sc, ax=ax[0])

    # Visualisation des pressions prédites
    sc = ax[1].scatter(x_coords, y_coords, c=predicted_pressures, cmap='viridis', s=10)
    ax[1].set_title("Pression prédite")
    fig.colorbar(sc, ax=ax[1])

    plt.show()

# --- Visualisation d'un exemple de test ---
def visualize_test_result(model, test_loader, test_params):
    model.eval()
    with torch.no_grad():
        for batch, params in zip(test_loader, test_params):
            # Effectuer la prédiction
            out = model(batch)
            # Dénormaliser les pressions
            true_pressures = denormalize_data(batch.y.numpy(), params[2], params[3])
            predicted_pressures = denormalize_data(out.squeeze().numpy(), params[2], params[3])

            # Extraire les coordonnées des nœuds du graphe
            x_coords = batch.x[:, 0].numpy()
            y_coords = batch.x[:, 1].numpy()

            # Affichage des résultats
            plot_pressure_field(true_pressures, predicted_pressures, x_coords, y_coords)
            break  # On peut arrêter après un test, ou boucler sur d'autres tests

# --- Main ---
if __name__ == "__main__":
    num_train_graphs = 80
    num_test_graphs = 20
    num_points_train = 5000
    num_points_test = 5000

    train_graphs, test_graphs, train_params, test_params = generate_graphs(
        num_train_graphs, num_test_graphs, num_points_train, num_points_test
    )
    train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    num_layers = 4 #Nombre de convolution
    activation_fn = torch.relu
    model = PressureGNN(in_channels=2, hidden_channels=8, out_channels=1, num_layers=num_layers, activation_fn=activation_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Mise à jour de tous les matrices W
    loss_fn = nn.MSELoss()

    epochs = 200
    train_losses, test_r2_scores = train_gnn_with_validation(
        model, train_loader, test_loader, train_params, test_params, optimizer, loss_fn, epochs
    )

    train_mse, train_r2 = evaluate_gnn(model, train_loader, train_params)
    test_mse, test_r2 = evaluate_gnn(model, test_loader, test_params)

    print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")

    # Tracer les courbes de perte et de R²
    plot_training_and_validation(train_losses, test_r2_scores)
        # Visualisation d'un résultat de test
    visualize_test_result(model, test_loader, test_params)
