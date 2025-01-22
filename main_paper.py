import clip
import os
import sys
import torch
import joblib
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from torchvision import datasets, transforms
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score


import cifar100_data as cd

parser = argparse.ArgumentParser(description="CLIP-based clustering and classification.")
parser.add_argument("--output_folder", type=str, default="results", help="Folder to save outputs.")
parser.add_argument("--reuse_embeddings", action="store_true", help="Reuse previously saved embeddings.")
parser.add_argument("--recompute_stats", action="store_true", help="Recompute statistics and reports from saved embeddings.")
args = parser.parse_args()

output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)


# Initialize logging to a file
log_file_path = os.path.join(output_folder, "results_log.txt")
log_file = open(log_file_path, "w")



class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, message):
        # Write to both console and the log file
        sys.__stdout__.write(message)  # Original stdout for console
        self.log_file.write(message)  # Write to log file

    def flush(self):
        sys.__stdout__.flush()
        self.log_file.flush()

# Redirect stdout to the Logger class
sys.stdout = Logger(log_file)

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP model loaded. Using device: {device}")

cifar10_dataset = datasets.CIFAR10(root="./cifar10", train=True, download=True)


# Dataset configuration
datasets_config = {
    "MNIST": {
        "dataset": datasets.MNIST,
        "transform": transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ]),
        "classes": [
            "This is a digit 0",
            "This is a digit 1",
            "This is a digit 2",
            "This is a digit 3",
            "This is a digit 4",
            "This is a digit 5",
            "This is a digit 6",
            "This is a digit 7",
            "This is a digit 8",
            "This is a digit 9",
            ],
        "classes_swapped": [
            "This is a digit 9",
            "This is a digit 0",
            "This is a digit 1",
            "This is a digit 2",
            "This is a digit 3",
            "This is a digit 4",
            "This is a digit 5",
            "This is a digit 6",
            "This is a digit 7",
            "This is a digit 8",
            ],
        "classes_grouped": [
            "This is a digit even",
            "This is a digit odd",
            "This is a digit even",
            "This is a digit odd",
            "This is a digit even",
            "This is a digit odd",
            "This is a digit even",
            "This is a digit odd",
            "This is a digit even",
            "This is a digit odd",
            ],
        "classes_hierarchical": [
            "This is a digit 0 even",
            "This is a digit 1 odd",
            "This is a digit 2 even",
            "This is a digit 3 odd",
            "This is a digit 4 even",
            "This is a digit 5 odd",
            "This is a digit 6 even",
            "This is a digit 7 odd",
            "This is a digit 8 even",
            "This is a digit 9 odd",
            ],
        "generic_prompt": "This is a digit.",
    },
    "CIFAR-10": {
        "dataset": datasets.CIFAR10,
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ]),
        "classes": [
            f"This is a airplane",
            f"This is a automobile",
            f"This is a bird",
            f"This is a cat",
            f"This is a deer",
            f"This is a dog",
            f"This is a frog",
            f"This is a horse",
            f"This is a ship",
            f"This is a truck",
            ],
        "classes_swapped": [
            f"This is a truck",
            f"This is a airplane",
            f"This is a automobile",
            f"This is a bird",
            f"This is a cat",
            f"This is a deer",
            f"This is a dog",
            f"This is a frog",
            f"This is a horse",
            f"This is a ship",
            ],
        "classes_grouped" : [
            f"This is a vehicle",
            f"This is a vehicle",
            f"This is a animal",
            f"This is a animal",
            f"This is a animal",
            f"This is a animal",
            f"This is a animal",
            f"This is a animal",
            f"This is a vehicle",
            f"This is a vehicle",
            ],
        "classes_hierarchical" : [
            f"This is a airplane vehicle",
            f"This is a automobile vehicle",
            f"This is a bird animal",
            f"This is a cat animal",
            f"This is a deer animal",
            f"This is a dog animal",
            f"This is a frog animal",
            f"This is a horse animal",
            f"This is a ship vehicle",
            f"This is a truck vehicle",
            ],
        "generic_prompt": "This is a thing.",
    }
}

for selected_index,\
    grouped_class,\
    hierarchical_class in zip(\
    cd.selected_indices,\
    cd.grouped_classes,\
    cd.hierarchical_classes,\
    ):
    dataset_config_x = cd.make_cifar_100_dataset(selected_index)
    dataset_config_x["classes_grouped"] = grouped_class
    dataset_config_x["classes_hierarchical"] = hierarchical_class
    datasets_config["CIFAR-100-" + "-".join([str(i) for i in selected_index])] = dataset_config_x



def save_clustering_results(output_path, clustering_object, cluster_assignments, cluster_centers):
    """
    Save clustering object, cluster assignments, and cluster centers to disk.
    
    Parameters:
        output_path: Path to save the clustering results.
        clustering_object: Trained clustering object (e.g., Spectral Clustering).
        cluster_assignments: Array of cluster assignments.
        cluster_centers: Array of cluster centers (if applicable).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(
        {
            "clustering_object": clustering_object,
            "cluster_assignments": cluster_assignments,
            "cluster_centers": cluster_centers,
        },
        output_path
    )
    print(f"Clustering results saved to {output_path}.")


def load_clustering_results(output_path):
    """
    Load clustering object, cluster assignments, and cluster centers from disk.
    
    Parameters:
        output_path: Path to load the clustering results.
    
    Returns:
        clustering_object, cluster_assignments, cluster_centers
    """
    if os.path.exists(output_path):
        results = joblib.load(output_path)
        print(f"Clustering results loaded from {output_path}.")
        return (
            results.get("clustering_object", None),
            results.get("cluster_assignments", None),
            results.get("cluster_centers", None),
        )
    else:
        return None, None, None


# Function to filter and load Cats-Dogs dataset from ImageNet structure
def load_cats_dogs_from_imagenet(config, split="train"):
    print(f"Loading Cats-Dogs dataset ({split}) from ImageNet structure...")
    
    # Build the list of valid class folders
    class_ids = config["categories"]["cat"] + config["categories"]["dog"]
    dataset_path = config["path"]
    
    # Filter only the folders for the desired class IDs
    filtered_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=config["transform"]
    )
    
    # Filter the dataset to include only images from the specified class IDs
    filtered_samples = [
        (path, label) for path, label in filtered_dataset.samples
        if os.path.basename(os.path.dirname(path)) in class_ids
    ]
    
    # Override the dataset's samples with the filtered ones
    filtered_dataset.samples = filtered_samples
    filtered_dataset.targets = [label for _, label in filtered_samples]

    # DataLoader
    data_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=64, shuffle=(split == "train"))
    print(f"Cats-Dogs dataset ({split}) loaded with {len(filtered_samples)} samples.")
    
    return filtered_dataset, data_loader

def load_dataset(name, config, split="train"):
    print(f"Loading {name} dataset ({split})...")
    if name == "Cats-Dogs":
        dataset = config["dataset"](root=config["path"], transform=config["transform"])
    else:
        dataset = config["dataset"](
            root=f"./{name.lower()}",
            train=(split == "train"),
            download=True,
            transform=config["transform"],
        )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=(split == "train"))
    print(f"{name} dataset ({split}) loaded and preprocessed.")
    return dataset, data_loader

def load_cifar100_subset(config, split="train"):
    """
    Load a subset of CIFAR-100 filtered by selected_indices and remap class labels.

    Parameters:
        config (dict): Configuration containing dataset, transform, selected_indices, and index_mapping.
        split (str): Whether to load the training or test split. Default is "train".

    Returns:
        tuple: Filtered dataset and DataLoader for the subset.
    """
    print(f"Loading CIFAR-100 subset ({split})...")

    # Load the subset dataset
    dataset = cd.SubsetCIFAR100(
        root="./cifar100",
        selected_indices=config["selected_indices"],
        index_mapping=config["index_mapping"],
        transform=config["transform"],
        train=(split == "train")
    )

    # Create a DataLoader for the subset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=(split == "train"))

    print(f"CIFAR-100 subset ({split}) loaded and preprocessed with {len(dataset)} samples.")
    return dataset, data_loader

def extract_embeddings(data_loader, prompts, dataset_name, prompt_type="task-specific", is_test=False):
    """
    Extract embeddings for a dataset, optimized for test sets by storing all prompt features
    and joint features in a single entry for each image.
    
    Parameters:
        data_loader: DataLoader object for the dataset.
        prompts: List of text prompts.
        dataset_name: Name of the dataset (e.g., "MNIST").
        prompt_type: Type of prompt being processed ("task-specific", "generic").
        is_test: Boolean indicating if the dataset is a test set. If True, include all prompts.
    
    Returns:
        List of embeddings with associated labels and features.
    """
    print(f"Encoding {prompt_type} text prompts for {dataset_name}...")
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().numpy()
    print(f"{prompt_type.capitalize()} text prompts encoded.")

    # Map unique prompts to indices
    unique_prompts = []
    for p in prompts:
        if p not in unique_prompts:
            unique_prompts.append(p)

    embeddings_data = []
    print(f"Extracting image embeddings for {dataset_name} ({prompt_type})...")
    for images, labels in tqdm(data_loader, desc="Processing images"):
        images = images.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images).cpu().numpy()

        for i, label in enumerate(labels):
            # Get the actual prompt for the given label
            corresponding_prompt = prompts[label.item()]
            # Find the index of this prompt in the unique prompts array
            #prompt_index = np.where(prompts == corresponding_prompt)[0][0]
            prompt_index = unique_prompts.index(corresponding_prompt)

            if is_test:
                # For test sets, store all prompt and joint features in a single entry
                all_joint_features = [
                    (image_features[i] + prompt_features) / 2 for prompt_features in text_features
                ]
                embeddings_data.append({
                    "label": label.item(),
                    "prompt": prompt_index,  # Index of the corresponding unique prompt
                    "image_features": image_features[i],
                    "text_features": text_features,  # Store all text features
                    "joint_features": all_joint_features  # Store all joint features
                })
            else:
                # For non-test sets, store only the true label's prompt and joint feature
                joint_feature = (image_features[i] + text_features[label]) / 2
                embeddings_data.append({
                    "label": label.item(),
                    "prompt": prompt_index,  # Index of the corresponding unique prompt
                    "image_features": image_features[i],
                    "text_features": text_features[label],
                    "joint_features": joint_feature
                })

    print(f"Embeddings extracted for {dataset_name} ({prompt_type}).")
    return embeddings_data


def map_clusters_to_labels_majority(
    cluster_assignments,
    train_embeddings,
    n_classes):

    n_clusters = n_classes*2
    # Map clusters to labels using majority voting
    cluster_to_label = {}
    label_counts = {label: 0 for label in range(n_classes)}

    for cluster in range(n_clusters):
        # Get all labels in the cluster
        cluster_labels = []
        for idx, data in enumerate(train_embeddings):
            if cluster_assignments[idx] == cluster:
                # Assigns the most common "prompt class" to the "idx" cluster
                # There can be less "prompt classes" than "label classes" (e.g. grouped)
                cluster_labels.append(data["prompt"])

        if cluster_labels:
            # Assign the most common "prompt class" to the cluster
            most_common_label = max(set(cluster_labels), key=cluster_labels.count)
            cluster_to_label[cluster] = most_common_label
            label_counts[most_common_label] += 1
        else:
            # Handle empty clusters (unlikely but possible)
            print(f"Warning: Cluster {cluster} is empty.")
            cluster_to_label[cluster] = None

    # Detect missing classes
    missing_classes = [cls for cls, count in label_counts.items() if count == 0]
    if missing_classes:
        print(f"Warning: The following classes are not assigned to any cluster: {missing_classes}")
    else:
        print("All classes are assigned to at least one cluster.")

    return cluster_to_label, missing_classes

# Perform clustering
def perform_clustering(embeddings_data, feature_key, n_clusters):
    """
    Perform clustering on the embeddings data using Spectral Clustering.

    Parameters:
        embeddings_data: List of embedding dictionaries containing features.
        feature_key: Key to access the feature array for clustering.
        n_clusters: Number of clusters to form.
    
    Returns:
        clustering_object: Trained Spectral Clustering object.
        cluster_assignments: Array of cluster assignments.
        cluster_centers: Approximated cluster centers as mean positions of points in each cluster.
    """
    n_clusters = n_clusters*2
    features_array = np.vstack([data[feature_key] for data in embeddings_data])

    # Perform Spectral Clustering
    clustering_object = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=42
    )
    cluster_assignments = clustering_object.fit_predict(features_array)

    # Compute approximate cluster centers
    cluster_centers = []
    for cluster in range(n_clusters):
        cluster_points = features_array[cluster_assignments == cluster]
        if len(cluster_points) > 0:
            cluster_center = np.mean(cluster_points, axis=0)
        else:
            cluster_center = np.zeros(features_array.shape[1])  # Handle empty clusters gracefully
        cluster_centers.append(cluster_center)

    cluster_centers = np.array(cluster_centers)

    return clustering_object, cluster_assignments, cluster_centers

# Perform classification
def classify_test_set(test_embeddings, cluster_centers, feature_key):
    predictions = []
    for data in test_embeddings:
        distances = np.linalg.norm(cluster_centers - data[feature_key], axis=1)
        predictions.append(np.argmin(distances))
    return predictions

# Visualize clusters with t-SNE
def visualize_clusters(
    embeddings_data,
    feature_key,
    dataset_name,
    prompt_type,
    output_path,
    prompt_labels=None,
    test_prompt_mode=False,
    selected_labels=None):
    """
    Visualize clusters using t-SNE.

    Parameters:
        embeddings_data: List of embedding dictionaries.
        feature_key: Key for accessing the feature array in embeddings.
        dataset_name: Name of the dataset.
        prompt_type: Type of prompts (e.g., "task-specific").
        output_path: Path to save the visualization.
        test_prompt_mode: If True, handle test embeddings with multiple joint features.
        selected_labels: Array indicating which joint embedding to select for each test sample.
    """
    print(f"Visualizing clusters for {feature_key} in {dataset_name} ({prompt_type})...")
    if test_prompt_mode:
        if selected_labels is None:
            raise ValueError("selected_labels must be provided in test_prompt_mode.")
        # Select the correct joint embedding for each test sample based on selected_labels
        features_array = np.vstack([
            data[feature_key][selected_labels[idx]]
            for idx, data in enumerate(embeddings_data)
        ])
    else:
        # Use standard feature array in non-test mode
        features_array = np.vstack([data[feature_key] for data in embeddings_data])

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features_array)

    plt.figure(figsize=(10, 8))
    correct_labels = [data["label"] for data in embeddings_data]
    

    if prompt_labels is None:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=correct_labels, cmap="tab10", s=10)
    else:
        scatter = plt.scatter(
            reduced_features[:, 0], 
            reduced_features[:, 1], 
            c=correct_labels, 
            cmap="tab10", 
            s=10
        )

        # Customize the colorbar to display prompts
        cbar = plt.colorbar(scatter, ticks=range(len(prompt_labels)))
        cbar.set_label("Prompts")
        # The last part (separated by whitespace) of the 'classes' prompts is the actual label
        yticklabels = [f"{p} ({c.split(' ')[-1]})" for p, c in zip(prompt_labels, config['classes'])]
        cbar.ax.set_yticklabels(yticklabels)

    #plt.colorbar(label="Classes")
    plt.title(f"t-SNE Visualization for {dataset_name} ({prompt_type})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()  # Prevent clipping of labels
    plt.savefig(output_path, bbox_inches='tight')  # Save figure with adjusted bounds
    
    print(f"Visualization saved: {output_path}")

def log_cluster_stats(cluster_assignments, train_embeddings, n_classes):
    print("Cluster statistics:")
    for cluster in range(n_classes):
        cluster_labels = [data["label"] for idx, data in enumerate(train_embeddings) if cluster_assignments[idx] == cluster]
        label_distribution = {label: cluster_labels.count(label) for label in set(cluster_labels)}
        print(f"Cluster {cluster}: {len(cluster_labels)} samples, Label distribution: {label_distribution}")

def calculate_cluster_purity(cluster_assignments, train_embeddings, cluster_to_label):
    total_samples = 0
    correct_samples = 0

    for cluster, assigned_label in cluster_to_label.items():
        cluster_labels = [data["label"] for idx, data in enumerate(train_embeddings) if cluster_assignments[idx] == cluster]
        if assigned_label is not None:
            correct_samples += cluster_labels.count(assigned_label)
        total_samples += len(cluster_labels)

    purity = correct_samples / total_samples if total_samples > 0 else 0
    print(f"Cluster purity: {purity:.4f}")
    return purity

def classify_test_set_with_majority_mapping(test_embeddings, cluster_centers, feature_key, cluster_to_label):
    predictions = []
    for data in test_embeddings:
        distances = np.linalg.norm(cluster_centers - data[feature_key], axis=1)
        closest_cluster = np.argmin(distances)
        predicted_label = cluster_to_label.get(closest_cluster, None)
        predictions.append(predicted_label)
    return predictions


def classify_test_images_with_prompts(
    test_embeddings,
    cluster_centers,
    cluster_to_label_mapping,
    feature_key="joint_features"
):
    """
    Classify test images by evaluating all precomputed joint features and
    assigning the label based on the closest valid cluster center.
    
    Parameters:
        test_embeddings: List of embeddings with precomputed joint features for all prompts.
        cluster_centers: Cluster centers from training data.
        cluster_to_label_mapping: Mapping of cluster indices to classes.
        feature_key: Key to access joint features in test_embeddings.
    
    Returns:
        predictions: List of predicted labels for test images.
    """
    print("Classifying test images with target-class and cluster-class alignment...")
    predictions = []
    
    for data in tqdm(test_embeddings, desc="Processing test images"):
        all_joint_features = data[feature_key]  # List of joint features for all prompts
        min_distance = float("inf")
        assigned_label = None

        for prompt_idx, joint_features in enumerate(all_joint_features):
            # Target class for the current joint embedding
            target_class = prompt_idx

            # Iterate over cluster centers and check for alignment
            for cluster_idx, cluster_center in enumerate(cluster_centers):
                cluster_class = cluster_to_label_mapping.get(cluster_idx)

                # Ensure alignment between target class and cluster class
                if cluster_class == target_class:
                    distance = np.linalg.norm(cluster_center - joint_features)

                    # Update assigned label if this is the closest valid cluster
                    if distance < min_distance:
                        min_distance = distance
                        assigned_label = target_class

        predictions.append(assigned_label)

    return predictions

# Main routine
results = {}
for dataset_name, config in datasets_config.items():
    print(f"Processing dataset: {dataset_name}")

    # Paths for saved embeddings
    # task-specific  embeddings
    train_embeddings_path = os.path.join(output_folder, f"{dataset_name}_train_embeddings.npy")
    test_embeddings_path = os.path.join(output_folder, f"{dataset_name}_test_embeddings.npy")
    # Hierarchical prompt embeddings
    train_hierarchical_embeddings_path =\
        os.path.join(output_folder, f"{dataset_name}_train_hierarchical_embeddings.npy")
    test_hierarchical_embeddings_path =\
        os.path.join(output_folder, f"{dataset_name}_test_hierarchical_embeddings.npy")
    # Grouped prompt embeddings
    train_grouped_embeddings_path =\
        os.path.join(output_folder, f"{dataset_name}_train_grouped_embeddings.npy")
    test_grouped_embeddings_path =\
        os.path.join(output_folder, f"{dataset_name}_test_grouped_embeddings.npy")
    # Generic prompt embeddings
    train_generic_embeddings_path = os.path.join(output_folder, f"{dataset_name}_train_generic_embeddings.npy")
    test_generic_embeddings_path = os.path.join(output_folder, f"{dataset_name}_test_generic_embeddings.npy")
    # Task-Swapped embeddings
    train_swapped_embeddings_path = os.path.join(output_folder, f"{dataset_name}_train_swapped_embeddings.npy")
    test_swapped_embeddings_path = os.path.join(output_folder, f"{dataset_name}_test_swapped_embeddings.npy")
    # Clustering paths
    task_specific_clustering_path = os.path.join(output_folder, f"{dataset_name}_task_specific_clustering.pkl")
    task_swapped_clustering_path = os.path.join(output_folder, f"{dataset_name}_swapped_clustering.pkl")
    task_hierarchical_clustering_path = os.path.join(
        output_folder, f"{dataset_name}_hierarchical_clustering.pkl")
    task_grouped_clustering_path = os.path.join(
        output_folder, f"{dataset_name}_grouped_clustering.pkl")
    generic_clustering_path = os.path.join(output_folder, f"{dataset_name}_generic_clustering.pkl")
    image_clustering_path = os.path.join(output_folder, f"{dataset_name}_image_clustering.pkl")


    # Load or compute embeddings
    if args.reuse_embeddings and os.path.exists(train_embeddings_path) and os.path.exists(test_embeddings_path):
        print(f"Reusing saved embeddings for {dataset_name}...")
        # Task-specific embeddings
        train_embeddings = np.load(train_embeddings_path, allow_pickle=True)
        test_embeddings = np.load(test_embeddings_path, allow_pickle=True)
        # Generic prompt embeddings
        train_generic_embeddings = np.load(train_generic_embeddings_path, allow_pickle=True)
        test_generic_embeddings = np.load(test_generic_embeddings_path, allow_pickle=True)
        # Task-Swapped embeddings
        train_swapped_embeddings = np.load(train_swapped_embeddings_path, allow_pickle=True)
        test_swapped_embeddings = np.load(test_swapped_embeddings_path, allow_pickle=True)
        # Task-Hierarchical embeddings
        train_hierarchical_embeddings = np.load(train_hierarchical_embeddings_path, allow_pickle=True)
        test_hierarchical_embeddings = np.load(test_hierarchical_embeddings_path, allow_pickle=True)
        # # Task-grouped embeddings
        train_grouped_embeddings = np.load(train_grouped_embeddings_path, allow_pickle=True)
        test_grouped_embeddings = np.load(test_grouped_embeddings_path, allow_pickle=True)
    else:
        if dataset_name.startswith("CIFAR-100"):
            train_dataset, train_loader = load_cifar100_subset(config, split="train")
            test_dataset, test_loader = load_cifar100_subset(config, split="test")
        else:
            train_dataset, train_loader = load_dataset(dataset_name, config, split="train")
            test_dataset, test_loader = load_dataset(dataset_name, config, split="test")

        # Extract embeddings
        train_embeddings = extract_embeddings(train_loader, config["classes"], dataset_name)
        test_embeddings = extract_embeddings(
            test_loader, config["classes"], dataset_name, prompt_type="test", is_test=True)

        # Generic embeddings
        train_generic_embeddings = extract_embeddings(
            train_loader, [config["generic_prompt"]] * len(config["classes"]),
            dataset_name, prompt_type="generic")
        test_generic_embeddings = extract_embeddings(
            test_loader, [config["generic_prompt"]] * len(config["classes"]),
            dataset_name, prompt_type="generic", is_test=True)

        # Swapped embeddings
        train_swapped_embeddings = extract_embeddings(
            train_loader, config["classes_swapped"], dataset_name, prompt_type="swapped")
        test_swapped_embeddings = extract_embeddings(
            test_loader, config["classes_swapped"], dataset_name, prompt_type="swapped",
            is_test=True)

        # Hierarchical embeddings
        train_hierarchical_embeddings = extract_embeddings(
            train_loader, config["classes_hierarchical"], dataset_name, prompt_type="hierarchical",)
        test_hierarchical_embeddings = extract_embeddings(
            test_loader, config["classes_hierarchical"], dataset_name, prompt_type="hierarchical",
            is_test=True)

        # Grouped embeddings
        train_grouped_embeddings = extract_embeddings(
            train_loader, config["classes_grouped"], dataset_name, prompt_type="grouped",)
        test_grouped_embeddings = extract_embeddings(
            test_loader, config["classes_grouped"], dataset_name, prompt_type="grouped",
            is_test=True)

        # Save embeddings
        np.save(train_embeddings_path, train_embeddings)
        np.save(test_embeddings_path, test_embeddings)
        np.save(train_generic_embeddings_path, train_generic_embeddings)
        np.save(test_generic_embeddings_path, test_generic_embeddings)
        np.save(train_swapped_embeddings_path, train_swapped_embeddings)
        np.save(test_swapped_embeddings_path, test_swapped_embeddings)
        np.save(train_hierarchical_embeddings_path, train_hierarchical_embeddings)
        np.save(test_hierarchical_embeddings_path, test_hierarchical_embeddings)
        np.save(train_grouped_embeddings_path, train_grouped_embeddings)
        np.save(test_grouped_embeddings_path, test_grouped_embeddings)
        print(f"Embeddings saved for {dataset_name}.")

    # SWAPPED LABELS
    print(f"\n\n-----STARTING SWAPPED-PROMPTS EXPERIMENTS FOR {dataset_name}-----\n\n")
    features_array_task_swapped = np.vstack(
        [data["joint_features"] for data in train_swapped_embeddings])
    labels_task_swapped = [data["prompt"] for data in train_swapped_embeddings]
    labels_test_task_swapped = [data["prompt"] for data in test_swapped_embeddings]

    # Load or compute clustering results
    spectral_task_swapped,\
    cluster_assignments_task_swapped,\
    cluster_centers_task_swapped = load_clustering_results(
        task_swapped_clustering_path
    )

    if spectral_task_swapped is None or cluster_assignments_task_swapped is None:
        spectral_task_swapped,\
        cluster_assignments_task_swapped,\
        cluster_centers_task_swapped = perform_clustering(
            train_swapped_embeddings, "joint_features", n_clusters=len(np.unique(config["classes_swapped"]))
        )
        save_clustering_results(
            task_swapped_clustering_path,
            spectral_task_swapped,
            cluster_assignments_task_swapped,
            cluster_centers_task_swapped
        )
    else:
        print(f"Using precomputed clustering results for {dataset_name} (task-swapped).")

    # Check for missing classes in task-specific prompts
    label_mapping_task_swapped, missing_classes_task_swapped =\
        map_clusters_to_labels_majority(
        cluster_assignments_task_swapped,
        train_swapped_embeddings,
        n_classes=len(np.unique(config["classes_swapped"]))
    )
    if missing_classes_task_swapped:
        print(f"Warning: Missing classes in task-swapped cluster mapping for {dataset_name}: {missing_classes_task_swapped}")

    # Evaluate clustering quality for task-specific prompts
    silhouette_task_swapped = silhouette_score(features_array_task_swapped, cluster_assignments_task_swapped)
    ari_task_swapped = adjusted_rand_score(labels_task_swapped, cluster_assignments_task_swapped)
    anmi_task_swapped = adjusted_mutual_info_score(labels_task_swapped, cluster_assignments_task_swapped)

    print(f"Task-swapped Silhouette Score: {silhouette_task_swapped:.4f}")
    print(f"Task-swapped ARI: {ari_task_swapped:.4f}")
    print(f"Task-swapped ANMI: {anmi_task_swapped:.4f}")

    # Perform classification for task-specific prompts
    predictions_task_swapped = classify_test_images_with_prompts(
        test_swapped_embeddings,
        cluster_centers_task_swapped,
        label_mapping_task_swapped,
        feature_key="joint_features"
    )
    accuracy_task_swapped = accuracy_score(
        labels_test_task_swapped, predictions_task_swapped)
    print(f"Classification accuracy for {dataset_name} (task-swapped prompts): {accuracy_task_swapped:.4f}")
    print(classification_report(labels_test_task_swapped, predictions_task_swapped))


    # GROUPED LABELS
    print(f"\n\n-----STARTING GROUPED EXPERIMENTS FOR {dataset_name}-----\n\n")

    features_array_task_grouped = np.vstack(
        [data["joint_features"] for data in train_grouped_embeddings])
    labels_task_grouped = [data["prompt"] for data in train_grouped_embeddings]
    labels_test_task_grouped = [data["prompt"] for data in test_grouped_embeddings]

    # Load or compute clustering results
    spectral_task_grouped,\
    cluster_assignments_task_grouped,\
    cluster_centers_task_grouped = load_clustering_results(
        task_grouped_clustering_path
    )

    if spectral_task_grouped is None or cluster_assignments_task_grouped is None:
        spectral_task_grouped,\
        cluster_assignments_task_grouped,\
        cluster_centers_task_grouped = perform_clustering(
            train_grouped_embeddings, "joint_features",
            n_clusters=len(np.unique(config["classes_grouped"]))
        )
        save_clustering_results(
            task_grouped_clustering_path,
            spectral_task_grouped,
            cluster_assignments_task_grouped,
            cluster_centers_task_grouped
        )
    else:
        print(f"Using precomputed clustering results for {dataset_name} (task-grouped).")

    # Check for missing classes in task-grouped prompts
    label_mapping_task_grouped, missing_classes_task_grouped =\
        map_clusters_to_labels_majority(
        cluster_assignments_task_grouped,
        train_grouped_embeddings,
        n_classes=len(np.unique(config["classes_grouped"]))
    )
    if missing_classes_task_grouped:
        print(f"Warning: Missing classes in task-grouped cluster mapping for {dataset_name}: {missing_classes_task_grouped}")

    # Evaluate clustering quality for task-grouped prompts
    silhouette_task_grouped = silhouette_score(features_array_task_grouped, cluster_assignments_task_grouped)
    ari_task_grouped = adjusted_rand_score(labels_task_grouped, cluster_assignments_task_grouped)
    anmi_task_grouped = adjusted_mutual_info_score(labels_task_grouped, cluster_assignments_task_grouped)

    print(f"Task-grouped Silhouette Score: {silhouette_task_grouped:.4f}")
    print(f"Task-grouped ARI: {ari_task_grouped:.4f}")
    print(f"Task-grouped ANMI: {anmi_task_grouped:.4f}")

    # Perform classification for task-grouped prompts
    predictions_task_grouped = classify_test_images_with_prompts(
        test_grouped_embeddings,
        cluster_centers_task_grouped,
        label_mapping_task_grouped,
        feature_key="joint_features"
    )
    accuracy_task_grouped = accuracy_score(
        labels_test_task_grouped, predictions_task_grouped)
    print(f"Classification accuracy for {dataset_name} (task-grouped prompts): {accuracy_task_grouped:.4f}")
    print(classification_report(labels_test_task_grouped, predictions_task_grouped))

    # TASK-SPECIFIC
    print(f"\n\n-----STARTING TASK-SPECIFIC EXPERIMENTS FOR {dataset_name}-----\n\n")
    features_array_task_specific = np.vstack([data["joint_features"] for data in train_embeddings])
    labels_task_specific = [data["prompt"] for data in train_embeddings]
    labels_test_task_specific = [data["prompt"] for data in test_embeddings]

    # Load or compute clustering results
    spectral_task_specific,\
    cluster_assignments_task_specific,\
    cluster_centers_task_specific = load_clustering_results(
        task_specific_clustering_path
    )

    if spectral_task_specific is None or cluster_assignments_task_specific is None:
        # Perform clustering with Spectral Clustering for task-specific prompts
        spectral_task_specific,\
        cluster_assignments_task_specific,\
        cluster_centers_task_specific = perform_clustering(
            train_embeddings, "joint_features", n_clusters=len(
                np.unique(config["classes"]))
        )
        save_clustering_results(
            task_specific_clustering_path,
            spectral_task_specific,
            cluster_assignments_task_specific,
            cluster_centers_task_specific
        )
    else:
        print(f"Using precomputed clustering results for {dataset_name} (task-specific).")

    # Check for missing classes in task-specific prompts
    label_mapping_task_specific, missing_classes_task_specific =\
        map_clusters_to_labels_majority(
        cluster_assignments_task_specific,
        train_embeddings,
        n_classes=len(np.unique(config["classes"]))
    )
    if missing_classes_task_specific:
        print(f"Warning: Missing classes in task-specific cluster mapping for {dataset_name}: {missing_classes_task_specific}")

    # Evaluate clustering quality for task-specific prompts
    silhouette_task_specific = silhouette_score(features_array_task_specific, cluster_assignments_task_specific)
    ari_task_specific = adjusted_rand_score(labels_task_specific, cluster_assignments_task_specific)
    anmi_task_specific = adjusted_mutual_info_score(labels_task_specific, cluster_assignments_task_specific)

    print(f"Task-Specific Silhouette Score: {silhouette_task_specific:.4f}")
    print(f"Task-Specific ARI: {ari_task_specific:.4f}")
    print(f"Task-Specific ANMI: {anmi_task_specific:.4f}")

    # Perform classification for task-specific prompts
    predictions_task_specific = classify_test_images_with_prompts(
        test_embeddings,
        cluster_centers_task_specific,
        label_mapping_task_specific,
        feature_key="joint_features"
    )
    accuracy_task_specific = accuracy_score(labels_test_task_specific, predictions_task_specific)
    print(f"Classification accuracy for {dataset_name} (task-specific prompts): {accuracy_task_specific:.4f}")
    print(classification_report(labels_test_task_specific, predictions_task_specific))

    # GENERIC # data['label'] to show that image embeddings and generic labels end up being the same
    print(f"\n\n-----STARTING GENERIC-PROMPT EXPERIMENTS FOR {dataset_name}-----\n\n")
    features_array_generic = np.vstack([data["joint_features"] for data in train_generic_embeddings])
    labels_generic = [data["label"] for data in train_generic_embeddings]
    labels_test_generic = [data["label"] for data in test_generic_embeddings]

    # Load or compute clustering results for generic prompts
    spectral_generic,\
    cluster_assignments_generic,\
    cluster_centers_generic = load_clustering_results(
        generic_clustering_path
    )

    if spectral_generic is None or cluster_assignments_generic is None:

        spectral_generic, cluster_assignments_generic,\
        cluster_centers_generic = perform_clustering(
            train_generic_embeddings, "joint_features", len(config["classes"])
        )
        save_clustering_results(
            generic_clustering_path,
            spectral_generic,
            cluster_assignments_generic,
            cluster_centers_generic)
    else:
        print(f"Using precomputed clustering results for {dataset_name} (generic prompts).")
    

    # Check for missing classes in generic prompts
    label_mapping_generic, missing_classes_generic = map_clusters_to_labels_majority(
        cluster_assignments_generic, train_generic_embeddings,
        n_classes=len(config["classes"])
    )
    if missing_classes_generic:
        print(f"Warning: Missing classes in generic cluster mapping for {dataset_name}: {missing_classes_generic}")

    # Evaluate clustering quality for generic prompts
    silhouette_generic = silhouette_score(features_array_generic, cluster_assignments_generic)
    ari_generic = adjusted_rand_score(labels_generic, cluster_assignments_generic)
    anmi_generic = adjusted_mutual_info_score(labels_generic, cluster_assignments_generic)

    print(f"Generic Silhouette Score: {silhouette_generic:.4f}")
    print(f"Generic ARI: {ari_generic:.4f}")
    print(f"Generic ANMI: {anmi_generic:.4f}")

    # Perform classification for generic prompts
    predictions_generic = classify_test_images_with_prompts(
        test_generic_embeddings,
        cluster_centers_generic,
        label_mapping_generic,
        feature_key="joint_features"
    )
    accuracy_generic = accuracy_score(labels_test_generic, predictions_generic)
    print(f"Generic Classification Accuracy: {accuracy_generic:.4f}")
    print(classification_report(labels_test_generic, predictions_generic))

    # ONLY IMAGES

    # Perform clustering on image embeddings
    print(f"\n\n-----STARTING IMAGES-ONLY EXPERIMENTS FOR {dataset_name}-----\n\n")
    features_array_image = np.vstack([data["image_features"] for data in train_embeddings])
    labels_image = [data["label"] for data in train_embeddings]

    # Load or compute clustering results for image embeddings
    spectral_image, cluster_assignments_image,\
    cluster_centers_image = load_clustering_results(image_clustering_path)

    if spectral_image is None or cluster_assignments_image is None:
        spectral_image, cluster_assignments_image, cluster_centers_image = perform_clustering(
            train_embeddings, "image_features", len(config["classes"])
        )
        save_clustering_results(
            image_clustering_path,
            spectral_image,
            cluster_assignments_image,
            cluster_centers_image)
    else:
        print(f"Using precomputed clustering results for {dataset_name} (image embeddings).")

    # Check for missing classes in image embeddings
    label_mapping_image, missing_classes_image = map_clusters_to_labels_majority(
        cluster_assignments_image, train_embeddings,
        n_classes=len(np.unique(config["classes"]))
    )
    if missing_classes_image:
        print(f"Warning: Missing classes in image cluster mapping for {dataset_name}: {missing_classes_image}")

    # Evaluate clustering quality for image embeddings
    silhouette_image = silhouette_score(features_array_image, cluster_assignments_image)
    ari_image = adjusted_rand_score(labels_image, cluster_assignments_image)
    anmi_image = adjusted_mutual_info_score(labels_image, cluster_assignments_image)

    print(f"Image Embeddings Silhouette Score: {silhouette_image:.4f}")
    print(f"Image Embeddings ARI: {ari_image:.4f}")
    print(f"Image Embeddings ANMI: {anmi_image:.4f}")

    # Perform classification for image embeddings
    predictions_image = classify_test_images_with_prompts(
        test_embeddings,
        cluster_centers_image,
        label_mapping_image,
        feature_key="image_features"
    )
    accuracy_image = accuracy_score(labels_test_task_specific, predictions_image)
    print(f"Image Embedding Classification Accuracy: {accuracy_image:.4f}")
    print(classification_report(labels_test_task_specific, predictions_image))


    # HIERARCHICAL LABELS
    print(f"\n\n-----STARTING HIERARCHICAL-PROMPTS EXPERIMENTS FOR {dataset_name}-----\n\n")
    features_array_task_hierarchical = np.vstack(
        [data["joint_features"] for data in train_hierarchical_embeddings])
    labels_task_hierarchical = [data["prompt"] for data in train_hierarchical_embeddings]
    labels_test_task_hierarchical = [data["prompt"] for data in test_hierarchical_embeddings]

    # Load or compute clustering results
    spectral_task_hierarchical,\
    cluster_assignments_task_hierarchical,\
    cluster_centers_task_hierarchical = load_clustering_results(
        task_hierarchical_clustering_path
    )

    if spectral_task_hierarchical is None or cluster_assignments_task_hierarchical is None:
        spectral_task_hierarchical,\
        cluster_assignments_task_hierarchical,\
        cluster_centers_task_hierarchical = perform_clustering(
            train_hierarchical_embeddings,
            "joint_features", n_clusters=len(np.unique(config["classes_hierarchical"]))
        )
        save_clustering_results(
            task_hierarchical_clustering_path,
            spectral_task_hierarchical,
            cluster_assignments_task_hierarchical,
            cluster_centers_task_hierarchical
        )
    else:
        print(f"Using precomputed clustering results for {dataset_name} (task-hierarchical).")

    # Check for missing classes in task-hierarchical prompts
    label_mapping_task_hierarchical, missing_classes_task_hierarchical =\
        map_clusters_to_labels_majority(
        cluster_assignments_task_hierarchical,
        train_hierarchical_embeddings,
        n_classes=len(np.unique(config["classes_hierarchical"]))
    )
    if missing_classes_task_hierarchical:
        print(f"Warning: Missing classes in task-hierarchical cluster mapping for {dataset_name}: {missing_classes_task_hierarchical}")

    # Evaluate clustering quality for task-hierarchical prompts
    silhouette_task_hierarchical = silhouette_score(features_array_task_hierarchical, cluster_assignments_task_hierarchical)
    ari_task_hierarchical = adjusted_rand_score(labels_task_hierarchical, cluster_assignments_task_hierarchical)
    anmi_task_hierarchical = adjusted_mutual_info_score(labels_task_hierarchical, cluster_assignments_task_hierarchical)

    print(f"Task-hierarchical Silhouette Score: {silhouette_task_hierarchical:.4f}")
    print(f"Task-hierarchical ARI: {ari_task_hierarchical:.4f}")
    print(f"Task-hierarchical ANMI: {anmi_task_hierarchical:.4f}")

    # Perform classification for task-hierarchical prompts
    predictions_task_hierarchical = classify_test_images_with_prompts(
        test_hierarchical_embeddings,
        cluster_centers_task_hierarchical,
        label_mapping_task_hierarchical,
        feature_key="joint_features"
    )
    accuracy_task_hierarchical = accuracy_score(
        labels_test_task_hierarchical, predictions_task_hierarchical)
    print(f"Classification accuracy for {dataset_name} (task-hierarchical prompts): {accuracy_task_hierarchical:.4f}")
    print(classification_report(labels_test_task_hierarchical, predictions_task_hierarchical))


    # Visualize clusters
    if args.recompute_stats:
        print(f"Generating t-SNE visualizations for {dataset_name}...")
        tsne_folder = os.path.join(output_folder, dataset_name, "tsne")
        os.makedirs(tsne_folder, exist_ok=True)

        # t-SNE for task-specific joint embeddings
        visualize_clusters(
            train_embeddings,
            "joint_features",
            dataset_name,
            "task-specific",
            os.path.join(tsne_folder, f"{dataset_name}_train_joint_tsne.png"),
            prompt_labels=config["classes"],
            )
        visualize_clusters(
            test_embeddings, "joint_features", dataset_name, "task-specific",
            os.path.join(tsne_folder, f"{dataset_name}_test_joint_tsne.png"),
            prompt_labels=config["classes"],
            test_prompt_mode=True,
            selected_labels=predictions_task_specific)

        # t-SNE for task-swapped joint embeddings
        visualize_clusters(train_swapped_embeddings, "joint_features", dataset_name, "task-swapped",
                           os.path.join(tsne_folder, f"{dataset_name}_train_swapped_joint_tsne.png"),
                           prompt_labels=config["classes_swapped"],)
        visualize_clusters(
            test_swapped_embeddings, "joint_features", dataset_name, "task-swapped",
            os.path.join(tsne_folder, f"{dataset_name}_test_swapped_joint_tsne.png"),
            prompt_labels=config["classes_swapped"],
            test_prompt_mode=True,
            selected_labels=predictions_task_swapped)

        # t-SNE for task-hierarchical joint embeddings
        visualize_clusters(train_hierarchical_embeddings, "joint_features", dataset_name, "task-hierarchical",
                           os.path.join(tsne_folder, f"{dataset_name}_train_hierarchical_joint_tsne.png"),
                           prompt_labels=config["classes_hierarchical"],)
        visualize_clusters(
            test_hierarchical_embeddings, "joint_features", dataset_name, "task-hierarchical",
            os.path.join(tsne_folder, f"{dataset_name}_test_hierarchical_joint_tsne.png"),
            prompt_labels=config["classes_hierarchical"],
            test_prompt_mode=True,
            selected_labels=predictions_task_hierarchical)

        # t-SNE for task-grouped joint embeddings
        visualize_clusters(train_grouped_embeddings, "joint_features", dataset_name, "task-grouped",
                           os.path.join(tsne_folder, f"{dataset_name}_train_grouped_joint_tsne.png"),
                           prompt_labels=config["classes_grouped"],)
        visualize_clusters(
            test_grouped_embeddings, "joint_features", dataset_name, "task-grouped",
            os.path.join(tsne_folder, f"{dataset_name}_test_grouped_joint_tsne.png"),
            prompt_labels=config["classes_grouped"],
            test_prompt_mode=True,
            selected_labels=predictions_task_grouped)

        # t-SNE for image embeddings
        visualize_clusters(train_embeddings, "image_features", dataset_name, "images",
                           os.path.join(tsne_folder, f"{dataset_name}_train_image_tsne.png"))
        visualize_clusters( 
            test_embeddings, "image_features", dataset_name, "images",
            os.path.join(tsne_folder, f"{dataset_name}_test_image_tsne.png"),
            test_prompt_mode=False, # Image only, without test prompts
            selected_labels=None)

        # t-SNE for joint features with generic prompts
        visualize_clusters(train_generic_embeddings, "joint_features", dataset_name, "generic_joint",
                           os.path.join(tsne_folder, f"{dataset_name}_train_generic_joint_tsne.png"))
        visualize_clusters(
            test_generic_embeddings, "joint_features", dataset_name, "generic_joint",
            os.path.join(tsne_folder, f"{dataset_name}_test_generic_joint_tsne.png"),
            test_prompt_mode=True,
            selected_labels=predictions_generic)



    # Store results
    results[dataset_name] = {
        "task_specific_accuracy": accuracy_task_specific,
        "task_specific_missing_classes": missing_classes_task_specific,
        "task_specific_silhouette": silhouette_task_specific,
        "task_specific_ARI": ari_task_specific,
        "task_specific_NMI": anmi_task_specific,

        "generic_accuracy": accuracy_generic,
        "generic_missing_classes": missing_classes_generic,
        "generic_silhouette": silhouette_generic,
        "generic_ARI": ari_generic,
        "generic_NMI": anmi_generic,

        "image_accuracy": accuracy_image,
        "image_missing_classes": missing_classes_image,
        "image_silhouette": silhouette_image,
        "image_ARI": ari_image,
        "image_NMI": anmi_image,
        
        "task_swapped_accuracy": accuracy_task_swapped,
        "task_swapped_missing_classes": missing_classes_task_swapped,
        "silhouette_task_swapped": silhouette_task_swapped,
        "task_swapped_ARI": ari_task_swapped,
        "task_swapped_NMI": anmi_task_swapped,

        "task_hierarchical_accuracy": accuracy_task_hierarchical,
        "task_hierarchical_missing_classes": missing_classes_task_hierarchical,
        "silhouette_task_hierarchical": silhouette_task_hierarchical,
        "task_hierarchical_ARI": ari_task_hierarchical,
        "task_hierarchical_NMI": anmi_task_hierarchical,

        "task_grouped_accuracy": accuracy_task_grouped,
        "task_grouped_missing_classes": missing_classes_task_grouped,
        "silhouette_task_grouped": silhouette_task_grouped,
        "task_grouped_ARI": ari_task_grouped,
        "task_grouped_NMI": anmi_task_grouped,
    }

# Save results
results_path = os.path.join(output_folder, "classification_results.npy")
np.save(results_path, results)
print(f"Results saved to {results_path}.")

# Save results in human-readable format (optional)
results_human_path = os.path.join(output_folder, "classification_results.txt")
with open(results_human_path, "w") as results_file:
    for dataset, metrics in results.items():
        results_file.write(f"Dataset: {dataset}\n")
        for metric, value in metrics.items():
            results_file.write(f"  {metric}: {value}\n")
        results_file.write("\n")
print(f"Human-readable results saved to {results_human_path}.")

# Close log file and restore stdout
log_file.close()
sys.stdout = sys.__stdout__