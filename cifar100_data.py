from torchvision import datasets, transforms
from torch.utils.data import Dataset

class SubsetCIFAR100(Dataset):
    def __init__(self, root, selected_indices, index_mapping, transform=None, train=True):
        """
        A dataset wrapper to filter CIFAR-100 classes and remap labels.

        Parameters:
            root (str): Path to the dataset.
            selected_indices (list): Indices of selected CIFAR-100 classes.
            index_mapping (dict): Mapping from CIFAR-100 indices to subset positions.
            transform (callable, optional): Transform to apply to the data.
            train (bool): Whether to load the training split.
        """
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
        self.selected_indices = set(selected_indices)
        self.index_mapping = index_mapping
        self.transform = transform

        # Filter dataset to include only the selected classes
        self.filtered_data = [
            (img, index_mapping[label]) for img, label in self.dataset
            if label in self.selected_indices
        ]

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        img, label = self.filtered_data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Predefined CIFAR-100 class names
CIFAR100_CLASSES = [
    "apple", #0
    "aquarium_fish", #1
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile", 
    "cup",
    "dinosaur", 
    "dolphin", 
    "elephant", 
    "flatfish",
    "forest", 
    "fox", 
    "girl", 
    "hamster", 
    "house",
    "kangaroo", 
    "keyboard", 
    "lamp", 
    "lawn_mower", 
    "leopard", 
    "lion", 
    "lizard", 
    "lobster", 
    "man",
    "maple_tree", 
    "motorcycle", 
    "mountain", 
    "mouse", 
    "mushroom", 
    "oak_tree", 
    "orange", 
    "orchid",
    "otter", 
    "palm_tree", 
    "pear", 
    "pickup_truck", 
    "pine_tree", 
    "plain",
    "plate", 
    "poppy", 
    "porcupine",
    "possum", 
    "rabbit", 
    "raccoon",
    "ray", 
    "road", 
    "rocket",
    "rose", 
    "sea", 
    "seal", 
    "shark", 
    "shrew",
    "skunk", 
    "skyscraper", 
    "snail", 
    "snake", 
    "spider", 
    "squirrel", 
    "streetcar", 
    "sunflower",
    "sweet_pepper", 
    "table", 
    "tank", 
    "telephone", 
    "television", 
    "tiger", 
    "tractor", 
    "train", 
    "trout",
    "tulip", 
    "turtle", 
    "wardrobe", 
    "whale", 
    "willow_tree", 
    "wolf", 
    "woman", 
    "worm"
]

def make_cifar_100_dataset(selected_indices):
    """
    Create a CIFAR-100 dataset configuration for a specific set of selected classes.

    Parameters:
        selected_indices (list): List of class indices for the selected CIFAR-100 classes.

    Returns:
        dict: Dataset configuration with transforms, classes, groups, and prompts.
    """
    # Load CIFAR-100 class names

    # Ensure the selected indices are valid
    selected_classes = [CIFAR100_CLASSES[idx] for idx in selected_indices]

    # Create swapped classes (rotate classes by one)
    swapped_classes = selected_classes[1:] + selected_classes[:1]

    # Transform for CIFAR-100
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # Map CIFAR-100 indices to subset positions
    index_mapping = {original: subset for subset, original in enumerate(selected_indices)}

    # Create the dataset configuration
    return {
        "dataset": datasets.CIFAR100,
        "transform": transform,
        "classes": [f"This is a {cls}" for cls in selected_classes],
        "classes_swapped": [f"This is a {cls}" for cls in swapped_classes],
        "classes_grouped": None,  # Placeholder for grouped classes (to be added manually)
        "classes_hierarchical": None,  # Placeholder for hierarchical classes (to be added manually)
        "generic_prompt": "This is a thing.",
        "selected_indices": selected_indices,
        "index_mapping": index_mapping
    }


# Selected classes: ['bicycle', 'bottle', 'chair', 'clock', 'couch', 'bear', 'chimpanzee', 'dolphin', 'elephant', 'flatfish']
selected_indices_1 = [8, 9, 20, 22, 25, 3, 21, 30, 31, 32]

grouped_classes_1 = [
    "This is an object", "This is an object", "This is an object",
    "This is an object", "This is an object", "This is an animal",
    "This is an animal", "This is an animal",
    "This is an animal", "This is an animal"
]

hierarchical_classes_1 = [
    "This is a bicycle object", "This is a bottle object", "This is a chair object",
    "This is a clock object", "This is a couch object", "This is a bear animal",
    "This is a chimpanzee animal", "This is a dolphin animal", "This is an elephant animal",
    "This is a flatfish animal"
]

# Selected classes: ['bus', 'motorcycle', 'pickup_truck', 'rocket', 'train', 'oak_tree', 'maple_tree', 'palm_tree', 'willow_tree', 'pine_tree']
selected_indices_2 = [13, 48, 58, 69, 90, 52, 47, 56, 96, 59]

grouped_classes_2 = [
    "This is a vehicle", "This is a vehicle", "This is a vehicle",
    "This is a vehicle", "This is a vehicle", "This is a plant",
    "This is a plant", "This is a plant", "This is a plant",
    "This is a plant"
]

hierarchical_classes_2 = [
    "This is a bus vehicle", "This is a motorcycle vehicle", "This is a pickup_truck vehicle",
    "This is a rocket vehicle", "This is a train vehicle", "This is an oak_tree plant",
    "This is a maple_tree plant", "This is a palm_tree plant", "This is a willow_tree plant",
    "This is a pine_tree plant"
]

# Selected classes: ['aquarium_fish', 'crab', 'dolphin', 'flatfish', 'lobster', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']
selected_indices_3 = [1, 26, 30, 32, 45, 6, 7, 14, 18, 24]

grouped_classes_3 = [
    "This is an aquatic animal", "This is an aquatic animal", "This is an aquatic animal",
    "This is an aquatic animal", "This is an aquatic animal", "This is an insect",
    "This is an insect", "This is an insect", "This is an insect",
    "This is an insect"
]

hierarchical_classes_3 = [
    "This is an aquarium_fish aquatic animal", "This is a crab aquatic animal", "This is a dolphin aquatic animal",
    "This is a flatfish aquatic animal", "This is a lobster aquatic animal", "This is a bee insect",
    "This is a beetle insect", "This is a butterfly insect", "This is a caterpillar insect",
    "This is a cockroach insect"
]

# Selected classes: ['bear', 'fox', 'elephant', 'lion', 'tiger', 'crocodile', 'lizard', 'snake', 'turtle', 'dinosaur']
selected_indices_4 = [3, 34, 31, 43, 88, 27, 44, 78, 93, 29]

grouped_classes_4 = [
    "This is a mammal", "This is a mammal", "This is a mammal",
    "This is a mammal", "This is a mammal", "This is a reptile",
    "This is a reptile", "This is a reptile", "This is a reptile",
    "This is a reptile"
]

hierarchical_classes_4 = [
    "This is a bear mammal", "This is a fox mammal", "This is an elephant mammal",
    "This is a lion mammal", "This is a tiger mammal", "This is a crocodile reptile",
    "This is a lizard reptile", "This is a snake reptile", "This is a turtle reptile",
    "This is a dinosaur reptile"
]

# Selected classes: ['rabbit', 'raccoon', 'ray', 'bear', 'fox', 'oak_tree', 'maple_tree', 'palm_tree', 'willow_tree', 'pine_tree']
selected_indices_5 = [65, 66, 67, 3, 34, 52, 47, 56, 96, 59]

grouped_classes_5 = [
    "This is an animal", "This is an animal", "This is an animal",
    "This is an animal", "This is an animal", "This is a plant",
    "This is a plant", "This is a plant", "This is a plant",
    "This is a plant"
]

hierarchical_classes_5 = [
    "This is a rabbit animal", "This is a raccoon animal", "This is a ray animal",
    "This is a bear animal", "This is a fox animal", "This is an oak_tree plant",
    "This is a maple_tree plant", "This is a palm_tree plant", "This is a willow_tree plant",
    "This is a pine_tree plant"
]


selected_indices = [
    selected_indices_1,
    selected_indices_2,
    selected_indices_3,
    selected_indices_4,
    selected_indices_5,
]

grouped_classes = [
    grouped_classes_1,
    grouped_classes_2,
    grouped_classes_3,
    grouped_classes_4,
    grouped_classes_5,
]

hierarchical_classes = [
    hierarchical_classes_1,
    hierarchical_classes_2,
    hierarchical_classes_3,
    hierarchical_classes_4,
    hierarchical_classes_5,
]