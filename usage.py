import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 1.
# ============================================================


training_records = [
    {"sequence": "A",     "mass": 100.0, "gt": 1.0},
    {"sequence": "AA",    "mass": 200.0, "gt": 2.0},
    {"sequence": "AAB",   "mass": 250.0, "gt": 2.5},
    {"sequence": "AABA",  "mass": 350.0, "gt": 3.5},
    {"sequence": "AABAC", "mass": 500.0, "gt": 5.0},
]


# ============================================================
# 2. 
# ============================================================

def build_transitions(records):


    transitions = []

    for i in range(1, len(records)):

        previous = records[i - 1]
        current = records[i]

        previous_sequence = previous["sequence"]
        current_sequence = current["sequence"]

        # Confirm that sequence really grew by one monomer
        if not current_sequence.staswith(previous_sequence):
            continue

        if len(current_sequence) != len(previous_sequence) + 1:
            continue

        added_monomer = current_sequence[-1]

        prev_mass = previous["mass"]
        prev_gt = previous["gt"]

        curr_mass = current["mass"]
        curr_gt = current["gt"]

        transitions.append({
            "previous_sequence": previous_sequence,
            "current_sequence": current_sequence,

            "prev_mass": prev_mass,
            "prev_gt": prev_gt,

            "curr_mass": curr_mass,
            "curr_gt": curr_gt,

            "delta_mass": curr_mass - prev_mass,
            "delta_gt": curr_gt - prev_gt,

            "label": added_monomer
        })

    return transitions


transitions = build_transitions(training_records)


print("\nTRAINING TRANSITIONS")
print("=" * 70)

for t in transitions:

    print(
        f"{t['previous_sequence']:8s}"
        f" -> "
        f"{t['current_sequence']:8s}"
        f" added={t['label']} "
        f" Δmass={t['delta_mass']:.3f}"
        f" Δgt={t['delta_gt']:.3f}"
    )


# ============================================================
# 3. 
# ============================================================

unique_labels = sorted(
    set(t["label"] for t in transitions)
)

label_to_id = {
    label: i
    for i, label in enumerate(unique_labels)
}

id_to_label = {
    i: label
    for label, i in label_to_id.items()
}


print("\nMonomers:")
print(label_to_id)


# ============================================================
# 4. 
# ============================================================


def transition_to_features(t):

    return np.array([
        t["prev_mass"],
        t["prev_gt"],

        t["curr_mass"],
        t["curr_gt"],

        t["delta_mass"],
        t["delta_gt"]

    ], dtype=np.float32)


X = np.array([
    transition_to_features(t)
    for t in transitions
])

y = np.array([
    label_to_id[t["label"]]
    for t in transitions
], dtype=np.int64)


# ============================================================
# 5.
# ============================================================

feature_mean = X.mean(axis=0)
feature_std = X.std(axis=0)

feature_std[
    feature_std < 1e-8
] = 1.0

X_scaled = (
    X - feature_mean
) / feature_std


# ============================================================
# 6.
# ============================================================

class TransitionDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        return (
            self.X[index],
            self.y[index]
        )


dataset = TransitionDataset(
    X_scaled,
    y
)

loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=True
)


# ============================================================
# 7. 
# ============================================================

class MetricAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=32,
        latent_dim=4
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                latent_dim
            )
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        reconstruction = self.decoder(z)

        return reconstruction, z


# ============================================================
# 8. 
# ============================================================

class SupervisedContrastiveLoss(nn.Module):

    def __init__(
        self,
        temperature=0.1
    ):

        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        embeddings,
        labels
    ):

        z = F.normalize(
            embeddings,
            dim=1
        )

        similarity = (
            z @ z.T
        ) / self.temperature

        n = len(labels)

        self_mask = torch.eye(
            n,
            dtype=torch.bool,
            device=embeddings.device
        )

        same_label = (
            labels.unsqueeze(0)
            ==
            labels.unsqueeze(1)
        )

        positive_mask = (
            same_label
            &
            ~self_mask
        )

        logits = similarity - (
            similarity.max(
                dim=1,
                keepdim=True
            ).values.detach()
        )

        exp_logits = (
            torch.exp(logits)
            *
            (~self_mask)
        )

        log_probability = (
            logits
            -
            torch.log(
                exp_logits.sum(
                    dim=1,
                    keepdim=True
                )
                +
                1e-8
            )
        )

        number_positive = (
            positive_mask.sum(dim=1)
        )

        valid = (
            number_positive > 0
        )

        # If a training batch contains no positive pair
        if not valid.any():

            return embeddings.sum() * 0.0

        positive_log_probability = (

            (
                positive_mask.float()
                *
                log_probability
            ).sum(dim=1)

            /

            number_positive.clamp(min=1)
        )

        return (
            -positive_log_probability[valid].mean()
        )


# ============================================================
# 9.
# ============================================================

model = MetricAutoencoder(
    input_dim=X.shape[1],
    hidden_dim=32,
    latent_dim=4
).to(DEVICE)

reconstruction_loss_function = nn.MSELoss()

contrastive_loss_function = (
    SupervisedContrastiveLoss(
        temperature=0.1
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. 
# ============================================================

EPOCHS = 1000

LAMBDA_METRIC = 0.5


print("\nTRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    model.train()

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        reconstructed, z = model(batch_x)

        reconstruction_loss = (
            reconstruction_loss_function(
                reconstructed,
                batch_x
            )
        )

        contrastive_loss = (
            contrastive_loss_function(
                z,
                batch_y
            )
        )

        loss = (
            reconstruction_loss
            +
            LAMBDA_METRIC
            *
            contrastive_loss
        )

        loss.backward()

        optimizer.step()

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} "
            f"total={loss.item():.4f} "
            f"recon={reconstruction_loss.item():.4f} "
            f"metric={contrastive_loss.item():.4f}"
        )


# ============================================================
# 11. 
# ============================================================


def calculate_latent_centroids(
    model,
    X_scaled,
    y
):

    model.eval()

    x_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        z = model.encoder(
            x_tensor
        )

        z = F.normalize(
            z,
            dim=1
        )

    centroids = {}

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
        device=DEVICE
    )

    for label_id in torch.unique(
        y_tensor
    ):

        mask = (
            y_tensor == label_id
        )

        centroid = (
            z[mask].mean(dim=0)
        )

        centroid = F.normalize(
            centroid.unsqueeze(0),
            dim=1
        ).squeeze(0)

        centroids[
            int(label_id.item())
        ] = centroid

    return centroids


centroids = calculate_latent_centroids(
    model,
    X_scaled,
    y
)


# ============================================================
# 12. 
# ============================================================


def calculate_monomer_statistics(
    transitions
):

    result = {}

    labels = sorted(
        set(
            t["label"]
            for t in transitions
        )
    )

    for label in labels:

        matching = [
            t
            for t in transitions
            if t["label"] == label
        ]

        delta_masses = np.array([
            t["delta_mass"]
            for t in matching
        ])

        delta_gts = np.array([
            t["delta_gt"]
            for t in matching
        ])

        result[label] = {

            "mass_mean":
                float(
                    delta_masses.mean()
                ),

            "mass_std":
                float(
                    delta_masses.std()
                ),

            "gt_mean":
                float(
                    delta_gts.mean()
                ),

            "gt_std":
                float(
                    delta_gts.std()
                )
        }

        # With tiny demonstration data,
        # std can be zero.
        #
        # Set minimum tolerances.
        #
        # THESE SHOULD BE TUNED FOR YOUR
        # REAL INSTRUMENT/DATA.
        #

        result[label]["mass_std"] = max(
            result[label]["mass_std"],
            0.5
        )

        result[label]["gt_std"] = max(
            result[label]["gt_std"],
            0.1
        )

    return result


monomer_stats = (
    calculate_monomer_statistics(
        transitions
    )
)


print("\nMONOMER PHYSICAL STATISTICS")
print("=" * 70)

for monomer, stats in monomer_stats.items():

    print(
        f"{monomer}: "
        f"mass={stats['mass_mean']:.3f} "
        f"+/- {stats['mass_std']:.3f}, "
        f"GT={stats['gt_mean']:.3f} "
        f"+/- {stats['gt_std']:.3f}"
    )


# ============================================================
# 13. 
# ============================================================

def make_candidate_features(
    previous_mass,
    previous_gt,
    current_mass,
    current_gt
):

    return np.array([

        previous_mass,
        previous_gt,

        current_mass,
        current_gt,

        current_mass - previous_mass,
        current_gt - previous_gt

    ], dtype=np.float32)


# ============================================================
# 14. 
# ============================================================

def encode_candidate(
    features
):

    scaled = (
        features - feature_mean
    ) / feature_std

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        z = model.encoder(
            tensor
        )

        z = F.normalize(
            z,
            dim=1
        )

    return z.squeeze(0)


# ============================================================
# 15.
# ============================================================

def latent_similarity(
    embedding,
    monomer
):

    label_id = (
        label_to_id[monomer]
    )

    centroid = (
        centroids[label_id]
    )

    similarity = (
        F.cosine_similarity(
            embedding.unsqueeze(0),
            centroid.unsqueeze(0)
        ).item()
    )

    # Cosine:
    #
    # -1 ... +1
    #
    # Convert to:
    #
    # 0 ... 1
    #

    similarity_01 = (
        similarity + 1
    ) / 2

    return similarity_01


# ============================================================
# 16. 
# ============================================================


def physical_mass_score(
    delta_mass,
    monomer
):

    expected = (
        monomer_stats[
            monomer
        ]["mass_mean"]
    )

    sigma = (
        monomer_stats[
            monomer
        ]["mass_std"]
    )

    error = (
        delta_mass
        -
        expected
    )

    score = np.exp(
        -0.5
        *
        (error / sigma) ** 2
    )

    return float(score)


# ============================================================
# 17. GT SCORE
# ============================================================

def gt_score(
    delta_gt,
    monomer
):

    expected = (
        monomer_stats[
            monomer
        ]["gt_mean"]
    )

    sigma = (
        monomer_stats[
            monomer
        ]["gt_std"]
    )

    error = (
        delta_gt
        -
        expected
    )

    score = np.exp(
        -0.5
        *
        (error / sigma) ** 2
    )

    return float(score)


# ============================================================
# 18.
# ============================================================
#


WEIGHT_LATENT = 0.35
WEIGHT_MASS = 0.50
WEIGHT_GT = 0.15


def score_transition_for_monomer(
    previous_mass,
    previous_gt,
    current_mass,
    current_gt,
    monomer
):

    delta_mass = (
        current_mass
        -
        previous_mass
    )

    delta_gt = (
        current_gt
        -
        previous_gt
    )

    features = make_candidate_features(
        previous_mass,
        previous_gt,
        current_mass,
        current_gt
    )

    embedding = encode_candidate(
        features
    )

    z_score = latent_similarity(
        embedding,
        monomer
    )

    m_score = physical_mass_score(
        delta_mass,
        monomer
    )

    r_score = gt_score(
        delta_gt,
        monomer
    )

    combined = (

        WEIGHT_LATENT
        *
        z_score

        +

        WEIGHT_MASS
        *
        m_score

        +

        WEIGHT_GT
        *
        r_score
    )

    return {

        "combined": float(combined),

        "latent": float(z_score),

        "mass": float(m_score),

        "gt": float(r_score),

        "delta_mass": float(delta_mass),

        "delta_gt": float(delta_gt)
    }


# ============================================================
# 19.
# ============================================================

def classify_transition(
    previous_mass,
    previous_gt,
    current_mass,
    current_gt
):

    all_scores = {}

    for monomer in unique_labels:

        all_scores[monomer] = (
            score_transition_for_monomer(

                previous_mass,
                previous_gt,

                current_mass,
                current_gt,

                monomer
            )
        )

    best_monomer = max(
        all_scores,
        key=lambda m:
            all_scores[m]["combined"]
    )

    return (
        best_monomer,
        all_scores[best_monomer],
        all_scores
    )


# ============================================================
# 20. 
# ============================================================


MIN_EDGE_SCORE = 0.55

# Gives some benefit to extending a plausible peptide path.
PATH_EXTENSION_REWARD = 0.40


def predict_peptide_path(
    peaks,
    initial_monomer=None
):
    """
    peaks:
        [
            (mass, GT),
            ...
        ]

    Returns best peptide-growth path.

    The peaks do NOT all need to belong
    to the peptide.
    """

    # Sort according to mass.
    # Depending on your chemistry,
    # you might want GT ordering instead.
    peaks = sorted(
        peaks,
        key=lambda x: x[0]
    )

    n = len(peaks)

    # best score ending at node j
    best_score = np.zeros(n)

    previous_node = np.full(
        n,
        -1,
        dtype=int
    )

    added_monomer = [
        None
        for _ in range(n)
    ]

    edge_details = [
        None
        for _ in range(n)
    ]

    # Number of peptide transitions
    # in path ending at node j.
    path_length = np.ones(
        n,
        dtype=int
    )

    for j in range(n):

        current_mass = peaks[j][0]
        current_gt = peaks[j][1]

        for i in range(j):

            previous_mass = peaks[i][0]
            previous_gt = peaks[i][1]

            # Peptide mass should increase
            if current_mass <= previous_mass:
                continue

            # Usually gt should not go backward
            # for this simplified example.
            #
            # Remove this constraint if that is
            # not biologically appropriate.
            if current_gt <= previous_gt:
                continue

            (
                monomer,
                best_edge,
                all_scores
            ) = classify_transition(

                previous_mass,
                previous_gt,

                current_mass,
                current_gt
            )

            edge_probability = (
                best_edge["combined"]
            )

            # Reject unlikely edges
            if (
                edge_probability
                <
                MIN_EDGE_SCORE
            ):
                continue

            # log() prevents blindly preferring
            # lots of mediocre transitions.
            #
            # extension reward favors a sequence
            # containing multiple strong edges.
            candidate_score = (

                best_score[i]

                +

                np.log(
                    edge_probability
                    +
                    1e-8
                )

                +

                PATH_EXTENSION_REWARD
            )

            if (
                candidate_score
                >
                best_score[j]
            ):

                best_score[j] = (
                    candidate_score
                )

                previous_node[j] = i

                added_monomer[j] = (
                    monomer
                )

                edge_details[j] = (
                    best_edge
                )

                path_length[j] = (
                    path_length[i] + 1
                )

    # Pick best terminal node.
    #
    # First favor paths with more transitions,
    # then score.
    #
    best_end = max(
        range(n),
        key=lambda i: (
            path_length[i],
            best_score[i]
        )
    )

    node_indices = []

    predicted_additions = []

    transition_details = []

    current = best_end

    while current != -1:

        node_indices.append(
            current
        )

        if (
            added_monomer[current]
            is not None
        ):

            predicted_additions.append(
                added_monomer[current]
            )

            transition_details.append({
                "from_index":
                    previous_node[current],

                "to_index":
                    current,

                "monomer":
                    added_monomer[current],

                "scores":
                    edge_details[current]
            })

        current = (
            previous_node[current]
        )

    node_indices.reverse()
    predicted_additions.reverse()
    transition_details.reverse()

    selected_peaks = [
        peaks[i]
        for i in node_indices
    ]

    # If first detected peptide species is known,
    # prepend it.
    if initial_monomer is not None:

        predicted_sequence = (
            initial_monomer
            +
            "".join(
                predicted_additions
            )
        )

    else:

        predicted_sequence = (
            "".join(
                predicted_additions
            )
        )

    selected_set = set(
        node_indices
    )

    noise_peaks = [
        peak
        for i, peak in enumerate(peaks)
        if i not in selected_set
    ]

    return {

        "sequence":
            predicted_sequence,

        "added_monomers":
            predicted_additions,

        "selected_peaks":
            selected_peaks,

        "noise_peaks":
            noise_peaks,

        "path_score":
            float(
                best_score[best_end]
            ),

        "transition_details":
            transition_details
    }


# ============================================================
# 21. 
# ============================================================


unknown_peaks = [

    (100.0, 1.0),

    (181.0, 1.4),   # noise

    (200.0, 2.0),

    (250.0, 2.5),

    (314.0, 3.1),   # noise

    (350.0, 3.5),

    (500.0, 5.0)
]


result = predict_peptide_path(
    unknown_peaks,

    # We know the first observed species is A.
    initial_monomer="A"
)


# ============================================================
# 22. 
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION")
print("=" * 70)

print(
    "Predicted sequence:",
    result["sequence"]
)

print(
    "\nSelected peptide peaks:"
)

for mass, gt in result["selected_peaks"]:

    print(
        f"mass={mass:.3f}, "
        f"GT={gt:.3f}"
    )


print(
    "\nRejected / noise peaks:"
)

for mass, gt in result["noise_peaks"]:

    print(
        f"mass={mass:.3f}, "
        f"GT={gt:.3f}"
    )


print(
    "\nTRANSITION DETAILS"
)

print("=" * 70)

for d in result[
    "transition_details"
]:

    from_peak = (
        unknown_peaks[
            d["from_index"]
        ]
    )

    to_peak = (
        unknown_peaks[
            d["to_index"]
        ]
    )

    s = d["scores"]

    print(
        f"\n{from_peak} -> {to_peak}"
    )

    print(
        f"Predicted monomer: "
        f"{d['monomer']}"
    )

    print(
        f"Δmass = "
        f"{s['delta_mass']:.3f}"
    )

    print(
        f"ΔGT = "
        f"{s['delta_gt']:.3f}"
    )

    print(
        f"latent score = "
        f"{s['latent']:.3f}"
    )

    print(
        f"mass score = "
        f"{s['mass']:.3f}"
    )

    print(
        f"GT score = "
        f"{s['gt']:.3f}"
    )

    print(
        f"combined score = "
        f"{s['combined']:.3f}"
    )
