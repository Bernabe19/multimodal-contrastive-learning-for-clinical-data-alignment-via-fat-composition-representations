import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import plotly.io as pio
pio.renderers.default = "browser"  # Use Plotly in browser
import plotly.express as px
# matplotlib.use('qt5agg')
from dotenv import load_dotenv

load_dotenv()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure derterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seed for dataloader workers
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

set_seed(42)
# --------- Multimodal Dataset ---------
class MultiModalDataset(Dataset):
    '''Order: fat, bioq, antro, muscle, demo, physio, labels'''
    def __init__(self, fat, bioq, antro, muscle, demo, meta, physio, labels):
        self.fat = fat
        self.bioq = bioq
        self.antro = antro
        self.muscle = muscle
        self.demo = demo
        self.meta = meta
        self.physio = physio
        self.labels = labels

    def __len__(self):
        return len(self.fat)

    def __getitem__(self, idx):
        return {
            'fat': self.fat[idx],
            'bioq': self.bioq[idx],
            'antro': self.antro[idx],
            'muscle': self.muscle[idx],
            'demo': self.demo[idx],
            'meta': self.meta[idx],
            'physio': self.physio[idx],
            'label': self.labels[idx]

        }

# --------- Encoder ---------
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim)
        )
        self.classifier = nn.Linear(out_dim, 3)  # For binary classification (not this case)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.classifier(z)
        return z, pred


def clip_style_contrastive_loss_7mod(embedding1, embedding2, embedding3, embedding4, embedding5, embedding6, embedding7, temperature=0.1, alpha=None, visceral_labels=None):
    """
    CLIP-style symmetric contrastive loss between Fat composition, Biochemical, Anthropometric, Muscle composition, Demographics, Metabolic, and Physiological embeddings.
    Calculates symmetric losses between each pair of views.
    embedding1: fat
    embedding2: bioq
    embedding3: antro
    embedding4: muscle
    embedding5: demo
    embedding6: meta
    embedding7: physio
    """
    contrastive_loss = None
    device = embedding1.device
    T = torch.exp(torch.tensor(temperature).to(device))
    # Normalization
    embedding1 = F.normalize(embedding1, dim=1)
    embedding2 = F.normalize(embedding2, dim=1)
    embedding3 = F.normalize(embedding3, dim=1)
    embedding4 = F.normalize(embedding4, dim=1)
    embedding5 = F.normalize(embedding5, dim=1)
    embedding6 = F.normalize(embedding6, dim=1)
    embedding7 = F.normalize(embedding7, dim=1)

    # Labels
    labels = torch.arange(embedding1.size(0)).to(device)
    # if visceral_labels is not None:
    #     labels = visceral_labels.to(device)

    def pairwise_contrastive(x, y):
        logits = torch.matmul(x, y.T) / T
        loss_x = F.cross_entropy(logits, labels)
        loss_y = F.cross_entropy(logits.T, labels)
        return (loss_x + loss_y) / 2

    # Fat-Modality
    loss_fb = pairwise_contrastive(embedding1, embedding2)
    loss_fa = pairwise_contrastive(embedding1, embedding3)
    loss_fm = pairwise_contrastive(embedding1, embedding4)
    loss_fd = pairwise_contrastive(embedding1, embedding5)
    loss_fme = pairwise_contrastive(embedding1, embedding6)
    loss_fp = pairwise_contrastive(embedding1, embedding7)

    # Modality-Modality
        # - Bioq - Modality
    loss_ba = pairwise_contrastive(embedding2, embedding3)
    loss_bm = pairwise_contrastive(embedding2, embedding4)
    loss_bd = pairwise_contrastive(embedding2, embedding5)
    loss_bme = pairwise_contrastive(embedding2, embedding6)
    loss_bp = pairwise_contrastive(embedding2, embedding7)
        # - Antrophometric - Modality
    loss_am = pairwise_contrastive(embedding3, embedding4)
    loss_ad = pairwise_contrastive(embedding3, embedding5)
    loss_ame = pairwise_contrastive(embedding3, embedding6)
    loss_ap = pairwise_contrastive(embedding3, embedding7)
        # - Muscle - Modality
    loss_md = pairwise_contrastive(embedding4, embedding5)
    loss_mme = pairwise_contrastive(embedding4, embedding6)
    loss_mp = pairwise_contrastive(embedding4, embedding7)
        # - Demo - Modality
    loss_dme = pairwise_contrastive(embedding5, embedding6)
    loss_dp = pairwise_contrastive(embedding5, embedding7)
        # - Meta - Modality
    loss_mep = pairwise_contrastive(embedding6, embedding7)     
        # - Physio - Modality (not necesary, previously calculated)

    # if visceral_labels is not None:
    #     labels = visceral_labels.to(device)


    anchor_loss = (loss_fb + loss_fa + loss_fm + loss_fd + loss_fme + loss_fp) / 6
    full_loss = (loss_fb + loss_fa + loss_fm + loss_fd + loss_fme + loss_fp + loss_ba + loss_bm + loss_bd + loss_bme + loss_bp + loss_am + loss_ad + loss_ame + loss_ap + loss_md + loss_mme + loss_mp + loss_dme + loss_dp + loss_mep) / 21
    contrastive_loss = alpha * anchor_loss + (1-alpha) * full_loss 

    return contrastive_loss


# --------- Training ---------
def train(model_dict, dataloader, test_dataloader, device='cpu', epochs=50, save=False, alpha=None):
    optimizer = torch.optim.SGD(all_params, lr=0.0001, weight_decay=1e-4, momentum=0.9)
    loss_history = []
    for epoch in range(epochs):
        model_dict = {k: v.train().to(device) for k, v in model_dict.items()}
        total_loss = 0

        for batch in dataloader:
            fat = batch['fat'].to(device).float()
            bioq = batch['bioq'].to(device).float()
            antro = batch['antro'].to(device).float()
            muscle = batch['muscle'].to(device).float()
            demo = batch['demo'].to(device).float()
            meta = batch['meta'].to(device).float()
            physio = batch['physio'].to(device).float()
            labels = batch['label'].to(device).long()
            

            z_fat, _ = model_dict['fat'](fat)
            z_bioq, _ = model_dict['bioq'](bioq)
            z_antro, _ = model_dict['antro'](antro)
            z_muscle, _ = model_dict['muscle'](muscle)
            z_demo, _ = model_dict['demo'](demo)
            z_meta, _ = model_dict['meta'](meta)
            z_physio, _ = model_dict['physio'](physio)

            loss = clip_style_contrastive_loss_7mod(z_fat, z_bioq, z_antro, z_muscle, z_demo, z_meta, z_physio, alpha=alpha, visceral_labels=labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")
    # Evaluation with test set
    if test_dataloader is not None:
        model_dict = {k: v.eval().to(device) for k, v in model_dict.items()}
        with torch.no_grad():
            test_accuracy_fat = 0
            test_accuracy_bioq = 0
            test_accuracy_antro = 0
            test_accuracy_muscle = 0
            test_accuracy_demo = 0
            test_accuracy_meta = 0
            test_accuracy_physio = 0
            test_accuracy = 0
            correct_fat = 0
            correct_bioq = 0
            correct_antro = 0
            correct_muscle = 0
            correct_demo = 0
            correct_meta = 0
            correct_physio = 0
            correct = 0
            total = 0
            for test_batch in test_dataloader:
                test_fat = test_batch['fat'].to(device).float()
                test_bioq = test_batch['bioq'].to(device).float()
                test_antro = test_batch['antro'].to(device).float()
                test_muscle = test_batch['muscle'].to(device).float()
                test_demo = test_batch['demo'].to(device).float()
                test_meta = test_batch['meta'].to(device).float()
                test_physio = test_batch['physio'].to(device).float()
                test_labels = test_batch['label'].to(device).long()

                _, pred_test_fat = model_dict['fat'](test_fat)
                _, pred_test_bioq = model_dict['bioq'](test_bioq)
                _, pred_test_antro = model_dict['antro'](test_antro)
                _, pred_test_muscle = model_dict['muscle'](test_muscle)
                _, pred_test_demo = model_dict['demo'](test_demo)
                _, pred_test_meta = model_dict['meta'](test_meta)
                _, pred_test_physio = model_dict['physio'](test_physio)
                accuracy_fat = (pred_test_fat.argmax(dim=1) == test_labels).sum().item()
                accuracy_bioq = (pred_test_bioq.argmax(dim=1) == test_labels).sum().item()
                accuracy_antro = (pred_test_antro.argmax(dim=1) == test_labels).sum().item()
                accuracy_muscle = (pred_test_muscle.argmax(dim=1) == test_labels).sum().item()
                accuracy_demo = (pred_test_demo.argmax(dim=1) == test_labels).sum().item()
                accuracy_meta = (pred_test_meta.argmax(dim=1) == test_labels).sum().item()
                accuracy_physio = (pred_test_physio.argmax(dim=1) == test_labels).sum().item()
                correct_fat += accuracy_fat
                correct_bioq += accuracy_bioq
                correct_antro += accuracy_antro
                correct_muscle += accuracy_muscle
                correct_demo += accuracy_demo
                correct_meta += accuracy_meta
                correct_physio += accuracy_physio

                total += test_labels.size(0)
            test_accuracy_fat = correct_fat / total
            test_accuracy_bioq = correct_bioq / total
            test_accuracy_antro = correct_antro / total
            test_accuracy_muscle = correct_muscle / total
            test_accuracy_demo = correct_demo / total
            test_accuracy_meta = correct_meta / total
            test_accuracy_physio = correct_physio / total
            correct = correct_fat + correct_bioq + correct_antro + correct_muscle + correct_demo + correct_meta + correct_physio
            print(f"Test Accuracy: {correct / (7 * total):.4f}")
            print(f"Test Accuracy Fat: {test_accuracy_fat:.4f}")
            print(f"Test Accuracy Bioq: {test_accuracy_bioq:.4f}")
            print(f"Test Accuracy Antro: {test_accuracy_antro:.4f}")
            print(f"Test Accuracy Muscle: {test_accuracy_muscle:.4f}")
            print(f"Test Accuracy Demo: {test_accuracy_demo:.4f}")
            print(f"Test Accuracy Meta: {test_accuracy_meta:.4f}")
            print(f"Test Accuracy Physio: {test_accuracy_physio:.4f}")


    # Save the model
    if save:
        torch.save(
            {k: v.state_dict() for k, v in model_dict.items()},
            os.path.join(model_path, "clip_tab_model.pth")
        )

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(figure_dir, "loss_training.png"))
    # plt.show()

def calculate_distances(fat_embeddings, bioq_embeddings, antro_embeddings, muscle_embeddings, demo_embeddings, meta_embeddings, physio_embeddings):
    cos = nn.CosineSimilarity(dim=1)
    cont = 0
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_4 = 0
    acc_5 = 0
    acc_6 = 0
    embed_size = len(fat_embeddings)
    print(len(fat_embeddings), len(bioq_embeddings), len(antro_embeddings), len(muscle_embeddings), len(demo_embeddings), len(meta_embeddings), len(physio_embeddings))
    for idx in range(len(fat_embeddings)):
        sim_fat_bioq = cos(fat_embeddings[idx], bioq_embeddings[idx])
        sim_fat_antro = cos(fat_embeddings[idx], antro_embeddings[idx])
        sim_fat_muscle = cos(fat_embeddings[idx], muscle_embeddings[idx])
        sim_fat_demo = cos(fat_embeddings[idx], demo_embeddings[idx])
        sim_fat_meta = cos(fat_embeddings[idx], meta_embeddings[idx])
        sim_fat_physio = cos(fat_embeddings[idx], physio_embeddings[idx])
        print(sim_fat_bioq, sim_fat_antro, sim_fat_muscle, sim_fat_demo, sim_fat_meta, sim_fat_physio, cont)
        acc_1 += sim_fat_bioq
        acc_2 += sim_fat_antro
        acc_3 += sim_fat_muscle
        acc_4 += sim_fat_demo
        acc_5 += sim_fat_meta
        acc_6 += sim_fat_physio
        cont += 1

    print(acc_1/embed_size, acc_2/embed_size, acc_3/embed_size, acc_4/embed_size, acc_5/embed_size, acc_6/embed_size)


# --------- Visualization with t-SNE ---------
def visualize_embeddings(model_dict, dataset, classes, device='cpu', test: list = None, filter=None, save=False, trained=True):
    model_dict = {k: v.eval().to(device) for k, v in model_dict.items()}
    fat_latents, bioq_latents, antro_latents, muscle_latents, demo_latents, meta_latents, physio_latents = [], [], [], [], [], [], []


    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            z_fat, _ = model_dict['fat'](sample['fat'].unsqueeze(0).to(device).float())
            z_bioq, _ = model_dict['bioq'](sample['bioq'].unsqueeze(0).to(device).float())
            z_antro, _ = model_dict['antro'](sample['antro'].unsqueeze(0).to(device).float())
            z_muscle, _ = model_dict['muscle'](sample['muscle'].unsqueeze(0).to(device).float())
            z_demo, _ = model_dict['demo'](sample['demo'].unsqueeze(0).to(device).float())
            z_meta, _ = model_dict['meta'](sample['meta'].unsqueeze(0).to(device).float())
            z_physio, _ = model_dict['physio'](sample['physio'].unsqueeze(0).to(device).float())

            fat_latents.append(z_fat)
            bioq_latents.append(z_bioq)
            antro_latents.append(z_antro)
            muscle_latents.append(z_muscle)
            demo_latents.append(z_demo)
            meta_latents.append(z_meta)
            physio_latents.append(z_physio)
    

    print(calculate_distances(fat_latents, bioq_latents, antro_latents, muscle_latents, demo_latents, meta_latents, physio_latents))
    test_fat_latents, test_bioq_latents, test_antro_latents, test_muscle_latents, test_demo_latents, test_meta_latents, test_physio_latents = [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(test[0])):
            z_fat_test, _ = model_dict['fat'](test[0][i].unsqueeze(0).to(device).float())
            z_bioq_test, _ = model_dict['bioq'](test[1][i].unsqueeze(0).to(device).float())
            z_antro_test, _ = model_dict['antro'](test[2][i].unsqueeze(0).to(device).float())
            z_muscle_test, _ = model_dict['muscle'](test[3][i].unsqueeze(0).to(device).float())
            z_demo_test, _ = model_dict['demo'](test[4][i].unsqueeze(0).to(device).float())
            z_meta_test, _ = model_dict['meta'](test[5][i].unsqueeze(0).to(device).float())
            z_physio_test, _ = model_dict['physio'](test[6][i].unsqueeze(0).to(device).float())

            test_fat_latents.append(z_fat_test)
            test_bioq_latents.append(z_bioq_test)
            test_antro_latents.append(z_antro_test)
            test_muscle_latents.append(z_muscle_test)
            test_demo_latents.append(z_demo_test)
            test_meta_latents.append(z_meta_test)
            test_physio_latents.append(z_physio_test)
            
    print("Working with test set:")
    print(calculate_distances(test_fat_latents, test_bioq_latents, test_antro_latents, test_muscle_latents, test_demo_latents, test_meta_latents, test_physio_latents))
    fat_latents = torch.cat(fat_latents).cpu().numpy()
    bioq_latents = torch.cat(bioq_latents).cpu().numpy()
    muscle_latents = torch.cat(muscle_latents).cpu().numpy()
    antro_latents = torch.cat(antro_latents).cpu().numpy()
    demo_latents = torch.cat(demo_latents).cpu().numpy()
    meta_latents = torch.cat(meta_latents).cpu().numpy()    
    physio_latents = torch.cat(physio_latents).cpu().numpy()

    test_fat_latents = torch.cat(test_fat_latents).cpu().numpy()
    test_bioq_latents = torch.cat(test_bioq_latents).cpu().numpy()
    test_muscle_latents = torch.cat(test_muscle_latents).cpu().numpy()
    test_antro_latents = torch.cat(test_antro_latents).cpu().numpy()
    test_demo_latents = torch.cat(test_demo_latents).cpu().numpy()
    test_meta_latents = torch.cat(test_meta_latents).cpu().numpy()
    test_physio_latents = torch.cat(test_physio_latents).cpu().numpy()

    all_latents = np.concatenate([fat_latents, bioq_latents, antro_latents, muscle_latents, demo_latents, meta_latents, physio_latents], axis=0)
    tsne = TSNE(n_components=2, random_state=42, metric="cosine")
    reduced = tsne.fit_transform(all_latents)
    modalities = ["Fat", "Bioq", "Antro", "Muscle", "Demo", "Meta", "Physio"]
    N = fat_latents.shape[0]  # Number of samples
    reduced = reduced.reshape(len(modalities), N, 2).transpose(1, 0, 2)

    if filter:
        reduced = reduced[np.array(filter)]
        classes = np.array(classes)[np.array(filter)]

    
    modality_colors = {"Fat": "red", "Bioq": "blue", "Antro": "green",
                    "Muscle": "black", "Demo": "orange", "Meta": "brown", "Physio": "pink"}
    class_markers = {0: 'o', 1: 's', 2: '^'}  

    plt.figure(figsize=(12, 8))
    for i in range(len(reduced)):
        coords = reduced[i]
        clase = classes[i]
        for j, mod in enumerate(modalities):
            plt.scatter(
                coords[j, 0], coords[j, 1],
                color=modality_colors[mod],
                marker=class_markers[clase],
                label=mod if i == 0 else "",
                alpha=0.7
            )
            plt.text(
                coords[j, 0] + 0.5,
                coords[j, 1],
                str(i),
                fontsize=8,
                color=modality_colors[mod],
                alpha=0.8
            )

    plt.title("t-SNE of Modalities with Sample Numbering (color = modality)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # eliminar duplicados
    plt.legend(by_label.values(), by_label.keys(), loc="best")
    plt.tight_layout()



    # Save representations
    if save and trained:
        # Training
        np.save(os.path.join(output_dir, "fat_latents.npy"), fat_latents)
        np.save(os.path.join(output_dir, "bioq_latents.npy"), bioq_latents)
        np.save(os.path.join(output_dir, "antro_latents.npy"), antro_latents)
        np.save(os.path.join(output_dir, "muscle_latents.npy"), muscle_latents)
        np.save(os.path.join(output_dir, "demo_latents.npy"), demo_latents)
        np.save(os.path.join(output_dir, "meta_latents.npy"), meta_latents)
        np.save(os.path.join(output_dir, "physio_latents.npy"), physio_latents)
        # Pruebas
        np.save(os.path.join(output_dir, "test_fat_latents.npy"), test_fat_latents)
        np.save(os.path.join(output_dir, "test_bioq_latents.npy"), test_bioq_latents)
        np.save(os.path.join(output_dir, "test_antro_latents.npy"), test_antro_latents)
        np.save(os.path.join(output_dir, "test_muscle_latents.npy"), test_muscle_latents)
        np.save(os.path.join(output_dir, "test_demo_latents.npy"), test_demo_latents)
        np.save(os.path.join(output_dir, "test_meta_latents.npy"), test_meta_latents)
        np.save(os.path.join(output_dir, "test_physio_latents.npy"), test_physio_latents)
        # Reduced
        np.save(os.path.join(output_dir, "reduced.npy"), reduced)
        # Save figure
        plt.savefig(os.path.join(figure_dir, "tsne_plot.png"))
    elif save and not trained:
        # Pre-Training
        np.save(os.path.join(output_dir, "pre_fat_latents.npy"), fat_latents)
        np.save(os.path.join(output_dir, "pre_bioq_latents.npy"), bioq_latents)
        np.save(os.path.join(output_dir, "pre_antro_latents.npy"), antro_latents)
        np.save(os.path.join(output_dir, "pre_muscle_latents.npy"), muscle_latents)
        np.save(os.path.join(output_dir, "pre_demo_latents.npy"), demo_latents)
        np.save(os.path.join(output_dir, "pre_meta_latents.npy"), meta_latents)
        np.save(os.path.join(output_dir, "pre_physio_latents.npy"), physio_latents)
        # Pre-Testing
        np.save(os.path.join(output_dir, "pre_test_fat_latents.npy"), test_fat_latents)
        np.save(os.path.join(output_dir, "pre_test_bioq_latents.npy"), test_bioq_latents)
        np.save(os.path.join(output_dir, "pre_test_antro_latents.npy"), test_antro_latents)
        np.save(os.path.join(output_dir, "pre_test_muscle_latents.npy"), test_muscle_latents)
        np.save(os.path.join(output_dir, "pre_test_demo_latents.npy"), test_demo_latents)
        np.save(os.path.join(output_dir, "pre_test_meta_latents.npy"), test_meta_latents)
        np.save(os.path.join(output_dir, "pre_test_physio_latents.npy"), test_physio_latents)
        # Pre-Reduced
        np.save(os.path.join(output_dir, "pre_reduced.npy"), reduced)
        # Save figure
        plt.savefig(os.path.join(figure_dir, "pre_tsne_plot.png"))

    # plt.show()
    


# --------- Execution ---------
if __name__ == '__main__':
    search = False
    # Load the data
    output_dir = os.getenv("EMBEDDINGS_PATH")
    figure_dir = os.getenv("FIGURES_PATH")
    data_path = os.getenv("DATA_PATH")
    model_path = os.getenv("MODELS_PATH")
    df = pd.read_csv(os.path.join(data_path, 'path/to/tabular.csv'))
    
    fat_data = df[["Visceral Fat", "Total Fat", "Fat Right Leg", "Fat Left Leg", "Fat Right Arm", "Fat Left Arm", "Fat Trunk"]]
    bioq_data = df[["Cholesterol", "Glucose"]]
    antro_data = df[["Height", "Weight", "Total Muscle","Wrist", "Waist", "Hip", "WHR"]]
    muscle_data = df[["Total Muscle", "Muscle Right Leg", "Muscle Left Leg", "Muscle Right Arm", "Muscle Left Arm", "Muscle Trunk"]]
    demo_data = df[["Age", "Gender"]]
    meta_data = df[["BMR", "TEE", "Activity"]] # "BMR" = Basal Metabolic Rate, "TEE" = Total Energy Expenditure, "Activity" = Activity Level
    physio_data = df[["Systolic", "Diastolic"]]

    # Instantiate scalers
    scaler_fat = StandardScaler()
    scaler_bioq = StandardScaler()
    scaler_antro = StandardScaler()
    scaler_muscle = StandardScaler()
    scaler_demo = StandardScaler()
    scaler_meta = StandardScaler()
    scaler_physio = StandardScaler()
    

    # Normalize data
    fat_data = np.array(scaler_fat.fit_transform(fat_data))#[0:20]
    fat_prueba = torch.tensor(fat_data[400:])
    fat_data = fat_data[:400]
    
    bioq_data = np.array(scaler_bioq.fit_transform(bioq_data))#[0:20]
    bioq_prueba = torch.tensor(bioq_data[400:])
    bioq_data = bioq_data[:400]
    
    antro_data = np.array(scaler_antro.fit_transform(antro_data))
    antro_prueba = torch.tensor(antro_data[400:])
    antro_data = antro_data[:400]

    muscle_data = np.array(scaler_muscle.fit_transform(muscle_data))
    muscle_prueba = torch.tensor(muscle_data[400:])
    muscle_data = muscle_data[:400]
    
    demo_data = np.array(scaler_demo.fit_transform(demo_data))
    demo_prueba = torch.tensor(demo_data[400:])
    mujer = (df["Gender"] == 1).values[:400]

    demo_data = demo_data[:400]

    meta_data = np.array(scaler_meta.fit_transform(meta_data))
    meta_prueba = torch.tensor(meta_data[400:])
    meta_data = meta_data[:400]

    physio_data = np.array(scaler_physio.fit_transform(physio_data))
    physio_prueba = torch.tensor(physio_data[400:])
    physio_data = physio_data[:400]


    Visceral_Fat = df["Visceral Fat"].values
    classes = []
    for fat in Visceral_Fat:
        if fat >= 0 and fat <= 9:
            classes.append(0)
        elif fat >= 10 and fat <= 14:
            classes.append(1)
        else:
            classes.append(2)

    test_classes = np.array(classes)[:400]
    classes = np.array(classes)[0:400]



    # classes = (Visceral_Fat >= 12).astype(int)[0:400]

    dataset = MultiModalDataset(
        torch.tensor(fat_data),
        torch.tensor(bioq_data),
        torch.tensor(antro_data),
        torch.tensor(muscle_data),
        torch.tensor(demo_data),
        torch.tensor(meta_data),
        torch.tensor(physio_data),
        torch.tensor(classes)

    )

    test_dataset = MultiModalDataset(
        fat_prueba,
        bioq_prueba,
        antro_prueba,
        muscle_prueba,
        demo_prueba,
        meta_prueba,
        physio_prueba,
        torch.tensor(test_classes)
    )
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_dict = {
        'fat': MLPEncoder(fat_data.shape[1]),
        'bioq': MLPEncoder(bioq_data.shape[1]),
        'antro': MLPEncoder(antro_data.shape[1]),
        'muscle': MLPEncoder(muscle_data.shape[1]),
        'demo': MLPEncoder(demo_data.shape[1]),
        'meta': MLPEncoder(meta_data.shape[1]),
        'physio': MLPEncoder(physio_data.shape[1]),
    }


    all_params = [p for model in model_dict.values() for p in model.parameters()]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not search:
        visualize_embeddings(model_dict, dataset, classes, device=device, test=[fat_prueba, bioq_prueba, antro_prueba, muscle_prueba, demo_prueba, meta_prueba, physio_prueba], filter=None, save=False, trained=False)
        train(model_dict, dataloader, test_dataloader, device=device, epochs=2000, save=False, alpha=0.9)
        visualize_embeddings(model_dict, dataset, classes, device=device, test=[fat_prueba, bioq_prueba, antro_prueba, muscle_prueba, demo_prueba, meta_prueba, physio_prueba], filter=None, save=False, trained=True)
    else:
        for a in np.arange(0.0, 1.0, 0.1):
            model_dict = {
                'fat': MLPEncoder(fat_data.shape[1]),
                'bioq': MLPEncoder(bioq_data.shape[1]),
                'antro': MLPEncoder(antro_data.shape[1]),
                'muscle': MLPEncoder(muscle_data.shape[1]),
                'demo': MLPEncoder(demo_data.shape[1]),
                'meta': MLPEncoder(meta_data.shape[1]),
                'physio': MLPEncoder(physio_data.shape[1]),
            }
            all_params = [p for model in model_dict.values() for p in model.parameters()]
            output_dir = os.getenv("EMBEDDINGS_PATH") + str(a)
            figure_dir = os.getenv("FIGURES_PATH") + str(a)
            model_path = os.getenv("MODELS_PATH") + str(a)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(figure_dir, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)
            print(f"Training with alpha = {a}")
            visualize_embeddings(model_dict, dataset, classes, device=device, test=[fat_prueba, bioq_prueba, antro_prueba, muscle_prueba, demo_prueba, meta_prueba, physio_prueba], filter=None, save=False, trained=False)
            train(model_dict, dataloader, test_dataloader, device=device, epochs=2000, save=False, alpha=a)
            visualize_embeddings(model_dict, dataset, classes, device=device, test=[fat_prueba, bioq_prueba, antro_prueba, muscle_prueba, demo_prueba, meta_prueba, physio_prueba], filter=None, save=False, trained=True)
    


       