import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
from rich import print
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib
from argparse import ArgumentParser
# Uncomment if you are on linux and want to use a specific backend
# matplotlib.use('qt5agg')

load_dotenv()


def load_clip_tab_data(file_path):
    """
    Load the CLIP tab data from a .npy file.
    
    Args:
        file_path (str): Path to the .npy file containing the CLIP tab data.
        
    Returns:
        np.ndarray: Loaded CLIP tab data.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def compute_average_cosine_similarity_matrix(modalities):
    acc = 0
    modality_names = list(modalities.keys())
    num_modalities = len(modality_names)
    cos = nn.CosineSimilarity(dim=1)
    matrix = np.zeros((num_modalities, num_modalities))

    for i, src_mod in enumerate(modality_names):
        src_embeddings = modalities[src_mod]
        for j, tgt_mod in enumerate(modality_names):
            tgt_embeddings = modalities[tgt_mod]
            # Calculate cosine similarity between all pairs
            sim = cos(src_embeddings, tgt_embeddings)
            matrix[i, j] = sim.mean().item()
            acc += matrix[i, j]
    prom_cos = acc / (num_modalities ** 2)
    print(f"Average cosine similarity: {prom_cos:.4f}")
    print(modality_names)
    return prom_cos, pd.DataFrame(matrix, index=modality_names, columns=modality_names)

def save_heatmap(df, save_path, title="", mask=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="viridis", mask=mask, fmt=".2f")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def retrieval_function(embeddings, query_embedding, top_k=5, method='cosine'):
    """
    Retrieve the top_k most similar embeddings to the query_embedding.
    

    Args:
        embeddings (np.ndarray): Array of embeddings to search in.
        query_embedding (np.ndarray): The embedding to compare against.
        top_k (int): Number of top similar embeddings to return.
        method (str): Similarity method to use ('cosine' or 'euclidean').
        
    Returns:
        list: Indices of the top_k most similar embeddings.
    """
    if method == 'cosine':
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(torch.tensor(embeddings), torch.tensor(query_embedding))
        top_indices = torch.topk(similarities, k=top_k).indices.numpy()        
        return top_indices
    
    elif method == 'euclidean':
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
        # Calculate the Euclidean distance (L2 norm) to each embedding
        distances = torch.norm(embeddings_tensor - query_tensor, dim=1)  # (N,)
        # Get indices of smallest distance (closest)
        top_distances, top_indices = torch.topk(-distances, k=top_k)  # invert for topk largest = top smallest distances
        return top_indices.numpy()

# function to calculate the MAE between two sets of characteristics of rows from a dataframe and a modality
def calculate_normalized_mae(df, row1_idx, row2_idx, columns=None):
    """
    Calcula el MAE, NMAE y MAE base entre dos filas de un DataFrame.

    Args:
        df (pd.DataFrame): Datos.
        row1_idx (int): Índice de la primera fila (predicción).
        row2_idx (int): Índice de la segunda fila (valor real).
        columns (list, optional): Columnas a comparar.

    Returns:
        dict: Contiene 'mae', 'nmae' y 'baseline_mae'.
    """
    if columns is not None:
        row1 = df.iloc[row1_idx][columns].astype(float)
        row2 = df.iloc[row2_idx][columns].astype(float)
    else:
        row1 = df.iloc[row1_idx].astype(float)
        row2 = df.iloc[row2_idx].astype(float)

    # MAE between rows
    mae = np.mean(np.abs(row1 - row2))

    # Trivial baseline: mean of the real values
    baseline = np.full_like(row2, row2.mean())
    baseline_mae = np.mean(np.abs(baseline - row2))

    # NMAE: normalized by the mean of the real values
    nmae = mae / row2.mean() if row2.mean() != 0 else np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs(row1 - row2) / np.abs(row2)
        relative_errors = relative_errors[~np.isnan(relative_errors) & ~np.isinf(relative_errors)]
        mre = np.mean(relative_errors) if len(relative_errors) > 0 else np.nan

    return {
        'mae': mae,
        # 'nmae': nmae,
        'mre': mre,
        # 'baseline_mae': baseline_mae
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_ret', type=int, default=4, help='Number of neighbors to retrieve (including the sample itself)')
    parser.add_argument('--method', type=str, default='cosine', choices=['cosine', 'euclidean'], help='Similarity method for retrieval')
    parser.add_argument('--modality', type=str, default='bioq', choices=['bioq', 'antro', 'muscle', 'demo', 'meta', 'physio'], help='Modality to impute')
    args = parser.parse_args()
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embed_dir = os.getenv("EMBEDDINGS_PATH")
    embed_dir_fix = os.path.join(ROOT_DIR, "embeddings", "exp10.1")
    figures_dir = os.getenv("FIGURES_PATH")
    data_path = os.path.join(ROOT_DIR, 'data')
    df = pd.read_csv(os.path.join(data_path, 'df_final3.csv'))
    save_files = False
    train_cos = []
    test_cos = []
    alpha = []

    antro_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "antro_latents.npy"))
    bioq_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "bioq_latents.npy"))
    demo_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "demo_latents.npy"))
    fat_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "fat_latents.npy"))
    meta_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "meta_latents.npy"))
    muscle_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "muscle_latents.npy"))
    physio_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "physio_latents.npy"))
    # Embedings from testing
    test_antro_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_antro_latents.npy"))
    test_bioq_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_bioq_latents.npy"))
    test_demo_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_demo_latents.npy"))
    test_fat_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_fat_latents.npy"))
    test_meta_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_meta_latents.npy"))
    test_muscle_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_muscle_latents.npy"))
    test_physio_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "test_physio_latents.npy"))

    pre_antro_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_antro_latents.npy"))
    pre_bioq_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_bioq_latents.npy"))
    pre_demo_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_demo_latents.npy"))
    pre_fat_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_fat_latents.npy"))
    pre_meta_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_meta_latents.npy"))
    pre_muscle_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_muscle_latents.npy"))
    pre_physio_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_physio_latents.npy"))
    # Embedings from testing
    pre_test_antro_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_antro_latents.npy"))
    pre_test_bioq_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_bioq_latents.npy"))
    pre_test_demo_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_demo_latents.npy"))
    pre_test_fat_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_fat_latents.npy"))
    pre_test_meta_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_meta_latents.npy"))
    pre_test_muscle_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_muscle_latents.npy"))
    pre_test_physio_latents = load_clip_tab_data(os.path.join(embed_dir_fix, "pre_test_physio_latents.npy"))


    fat_data = df[["Visceral Fat", "Total Fat", "Fat Right Leg", "Fat Left Leg", "Fat Right Arm", "Fat Left Arm", "Fat Trunk"]]
    bioq_data = df[["Cholesterol", "Glucose"]]
    antro_data = df[["Height", "Weight", "Total Muscle","Wrist", "Waist", "Hip", "WHR"]]
    muscle_data = df[["Total Muscle", "Muscle Right Leg", "Muscle Left Leg", "Muscle Right Arm", "Muscle Left Arm", "Muscle Trunk"]]
    demo_data = df[["Age", "Gender"]]
    meta_data = df[["BMR", "TEE", "Activity"]] # "BMR" = Basal Metabolic Rate, "TEE" = Total Energy Expenditure, "Activity" = Activity Level
    physio_data = df[["Systolic", "Diastolic"]]

    fat_cols = ["Visceral Fat", "Total Fat", "Fat Right Leg", "Fat Left Leg", "Fat Right Arm", "Fat Left Arm", "Fat Trunk"]
    bioq_cols = ["Cholesterol", "Glucose"]
    antro_cols = ["Height", "Weight", "Total Muscle", "Wrist", "Waist", "Hip", "WHR"]
    muscle_cols = ["Total Muscle", "Muscle Right Leg", "Muscle Left Leg", "Muscle Right Arm", "Muscle Left Arm", "Muscle Trunk"]
    demo_cols = ["Age", "Gender"]
    meta_cols = ["BMR", "TEE", "Activity"]
    physio_cols = ["Systolic", "Diastolic"]

    all_modalities = list(set(
    fat_cols + bioq_cols + antro_cols + muscle_cols + demo_cols + meta_cols + physio_cols
    ))

    # Filter the original dataframe
    df = df[all_modalities]

    modality = args.modality
    embedding1 = np.concatenate([fat_latents, test_fat_latents], axis=0)
    embedding2 = np.concatenate([bioq_latents, test_bioq_latents], axis=0)
    embedding3 = np.concatenate([antro_latents, test_antro_latents], axis=0)
    embedding4 = np.concatenate([muscle_latents, test_muscle_latents], axis=0)
    embedding5 = np.concatenate([demo_latents, test_demo_latents], axis=0)
    embedding6 = np.concatenate([meta_latents, test_meta_latents], axis=0)
    embedding7 = np.concatenate([physio_latents, test_physio_latents], axis=0)
    embeddings = np.concatenate([embedding1, embedding2, embedding3, embedding4, embedding5, embedding6, embedding7], axis=1)
    if modality == 'bioq':
        embedding2 = np.zeros_like(embedding2)
    elif modality == 'antro':
        embedding3 = np.zeros_like(embedding3)
    elif modality == 'muscle':
        embedding4 = np.zeros_like(embedding4)
    elif modality == 'demo':
        embedding5 = np.zeros_like(embedding5)
    elif modality == 'meta':
        embedding6 = np.zeros_like(embedding6)
    elif modality == 'physio':
        embedding7 = np.zeros_like(embedding7)
    embeddings_test = np.concatenate([embedding1, embedding2, embedding3, embedding4, embedding5, embedding6, embedding7], axis=1)
    print("Shape of embeddings:", embeddings.shape)
    # embeddings = np.concatenate([fat_latents, test_fat_latents], axis=0)
    df_missing = df.copy()
    num_ret = args.num_ret
    test_index = 90
    acc_mae_knn = 0
    acc_mre_knn = 0
    acc_mae_knn_all = 0
    acc_mre_knn_all = 0
    acc_mae_mice = 0
    acc_mre_mice = 0
    acc_mae_mice_all = 0
    acc_mre_mice_all = 0
    acc_mae_ours = 0
    acc_mre_ours = 0

    mod_data = physio_data.columns.tolist()
    fat_data = fat_data.columns.tolist()
    true_values = df.loc[test_index, mod_data].copy()
    for test_index in range(len(embeddings)):
        df_missing = df.copy()
        true_values = df.loc[test_index, mod_data].copy()
        df_missing.loc[test_index, mod_data] = np.nan
  
        fat_matrix = df_missing[fat_data].copy()
        fat_matrix = fat_matrix.fillna(fat_matrix.mean())
       
        nn_model = NearestNeighbors(n_neighbors=num_ret)
        nn_model.fit(fat_matrix)

        
        test_sample = fat_matrix.loc[[test_index]]
        distances, indices = nn_model.kneighbors(test_sample)
      
        neighbors_idx = indices[0][indices[0] != test_index]
        mean_muscle_values = df.loc[neighbors_idx, mod_data].mean()

        df_manual_imputed = df_missing.copy()
        df_manual_imputed.loc[test_index, mod_data] = mean_muscle_values

        mae_knn = calculate_normalized_mae(
            df=pd.concat([df_manual_imputed, df.loc[[test_index]]]),
            row1_idx=test_index,
            row2_idx=len(df),
            columns=mod_data
        )

        df_subset = df[fat_data + mod_data].copy()

        true_values = df.loc[test_index, mod_data].copy()

        df_subset.loc[test_index, mod_data] = np.nan

        imputer = IterativeImputer(random_state=42)

        df_imputed_array = imputer.fit_transform(df_subset)
        df_imputed = pd.DataFrame(df_imputed_array, columns=df_subset.columns, index=df_subset.index)

        imputed_values = df_imputed.loc[test_index, mod_data]

        mae_mice = calculate_normalized_mae(
            df=pd.concat([df_imputed, df.loc[[test_index]]]),
            row1_idx=test_index,
            row2_idx=len(df),
            columns=mod_data
        )

        df_missing = df.copy()
        true_values = df.loc[test_index, mod_data].copy()

        df_missing.loc[test_index, mod_data] = np.nan

        all_features = [col for col in df.columns if col not in mod_data]
        feature_matrix = df_missing[all_features].copy()
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())
    
        nn_model = NearestNeighbors(n_neighbors=num_ret)
        nn_model.fit(feature_matrix)

        test_sample = feature_matrix.loc[test_index].values.reshape(1, -1)
        distances, indices = nn_model.kneighbors(test_sample)

        neighbors_idx = indices[0][indices[0] != test_index]

        mean_muscle_values = df.loc[neighbors_idx, mod_data].mean()

        df_manual_imputed = df_missing.copy()
        df_manual_imputed.loc[test_index, mod_data] = mean_muscle_values

        mae_knn_all = calculate_normalized_mae(
            df=pd.concat([df_manual_imputed, df.loc[[test_index]]]),
            row1_idx=test_index,
            row2_idx=len(df),
            columns=mod_data
        )

        df_full = df.copy()

        true_values = df_full.loc[test_index, mod_data].copy()

        df_full.loc[test_index, mod_data] = np.nan

        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_imputed_array = imputer.fit_transform(df_full)

        df_imputed = pd.DataFrame(df_imputed_array, columns=df_full.columns, index=df_full.index)

        mae_mice_all = calculate_normalized_mae(
            df=pd.concat([df_imputed, df.loc[[test_index]]]),
            row1_idx=test_index,
            row2_idx=len(df),
            columns=mod_data
        )

        query_embedding = embeddings_test[test_index]
        retrieved_indices = retrieval_function(embeddings=np.delete(embeddings, test_index, axis=0), 
                                            query_embedding=query_embedding, 
                                            top_k=num_ret, method=args.method)
        retrieved_indices = [idx if idx < test_index else idx + 1 for idx in retrieved_indices]
        retrieved_indices = retrieved_indices[1:]
        mean_values = df.loc[retrieved_indices, mod_data].mean()

        df_retrieval_manual = df_missing.copy()
        df_retrieval_manual.loc[test_index, mod_data] = mean_values

        mae_retrieval_manual = calculate_normalized_mae(
            df=pd.concat([df_retrieval_manual, df.loc[[test_index]]]),
            row1_idx=test_index,
            row2_idx=len(df),  
            columns=mod_data
        )
        acc_mae_knn += mae_knn['mae']
        acc_mre_knn += mae_knn['mre']
        acc_mae_mice += mae_mice['mae']
        acc_mre_mice += mae_mice['mre']
        acc_mae_knn_all += mae_knn_all['mae']
        acc_mre_knn_all += mae_knn_all['mre']
        acc_mae_mice_all += mae_mice_all['mae']
        acc_mre_mice_all += mae_mice_all['mre']
        acc_mae_ours += mae_retrieval_manual['mae']
        acc_mre_ours += mae_retrieval_manual['mre']

    print(test_index)
    print(neighbors_idx)
    print(retrieved_indices)
    print(f"Results: k=", num_ret - 1, f"{modality} -> ")
    print(f"MAE KNN ({modality}):", acc_mae_knn / len(embeddings))
    print(f"MRE KNN ({modality}):", acc_mre_knn / len(embeddings))
    print(f"MAE MICE ({modality}):", acc_mae_mice / len(embeddings))
    print(f"MRE MICE ({modality}):", acc_mre_mice / len(embeddings))
    print(f"MAE KNN (all):", acc_mae_knn_all / len(embeddings))
    print(f"MRE KNN (all):", acc_mre_knn_all / len(embeddings))
    print(f"MAE MICE (all):", acc_mae_mice_all / len(embeddings))
    print(f"MRE MICE (all):", acc_mre_mice_all /  len(embeddings))
    print(f"MAE Retrieval (embeddings):", acc_mae_ours / len(embeddings))
    print(f"MRE Retrieval (embeddings):", acc_mre_ours / len(embeddings))


            