import os
import re
import shutil
import random
from collections import defaultdict, Counter

SOURCE_DIR = "data/processed_v2"
OUTPUT_DIR = "data/split"

CLASSES = ["clear", "drive", "drop", "net", "smash"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def get_match_id(folder_name):
    match = re.search(r"MATCH\d+", folder_name.upper())
    return match.group(0) if match else None


print("=" * 60)
print("BADMINTON MATCH-LEVEL DATASET SPLITTING")
print("=" * 60)

match_class_sequences = defaultdict(lambda: defaultdict(list))

for class_name in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, class_name)

    folders = [
        f for f in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, f))
    ]

    print(f"{class_name.upper():10} : {len(folders)} sequences")

    for folder in folders:
        match_id = get_match_id(folder)

        if match_id is None:
            print(f"WARNING: Match ID not found: {folder}")
            continue

        match_class_sequences[match_id][class_name].append(folder)


matches = sorted(match_class_sequences.keys())

print(f"\nTotal unique matches: {len(matches)}")

# --------------------------------------------------
# Calculate class totals
# --------------------------------------------------

class_totals = Counter()

for match in matches:
    for class_name in CLASSES:
        class_totals[class_name] += len(
            match_class_sequences[match][class_name]
        )

print("\nTotal sequences by class:")
for class_name in CLASSES:
    print(f"{class_name.upper():10} : {class_totals[class_name]}")


# --------------------------------------------------
# Find a good match-level split
# --------------------------------------------------

target_val = {c: class_totals[c] * VAL_RATIO for c in CLASSES}
target_test = {c: class_totals[c] * TEST_RATIO for c in CLASSES}

best_score = float("inf")
best_split = None

for _ in range(20000):

    shuffled = matches.copy()
    random.shuffle(shuffled)

    n_test = max(1, round(len(matches) * TEST_RATIO))
    n_val = max(1, round(len(matches) * VAL_RATIO))

    test_matches = shuffled[:n_test]
    val_matches = shuffled[n_test:n_test + n_val]
    train_matches = shuffled[n_test + n_val:]

    def class_counts(match_list):
        counts = Counter()

        for match in match_list:
            for class_name in CLASSES:
                counts[class_name] += len(
                    match_class_sequences[match][class_name]
                )

        return counts

    val_counts = class_counts(val_matches)
    test_counts = class_counts(test_matches)

    score = 0

    for class_name in CLASSES:

        val_error = abs(
            val_counts[class_name] - target_val[class_name]
        )

        test_error = abs(
            test_counts[class_name] - target_test[class_name]
        )

        score += val_error + test_error

        # Strong penalty if validation has zero examples
        if val_counts[class_name] == 0:
            score += 1000

        # Strong penalty for extremely small validation classes
        if val_counts[class_name] < 0.08 * class_totals[class_name]:
            score += 300

    if score < best_score:
        best_score = score
        best_split = (
            train_matches,
            val_matches,
            test_matches,
        )


train_matches, val_matches, test_matches = best_split

print("\nMATCH SPLIT")
print("=" * 60)

print(f"Train matches      : {len(train_matches)}")
print(f"Validation matches : {len(val_matches)}")
print(f"Test matches       : {len(test_matches)}")

print("\nTrain:", sorted(train_matches))
print("Validation:", sorted(val_matches))
print("Test:", sorted(test_matches))


# --------------------------------------------------
# Copy sequences
# --------------------------------------------------

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for split_name in ["train", "val", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split_name, class_name),
            exist_ok=True
        )


def copy_split(match_list, split_name):

    total = 0

    for match in match_list:

        for class_name in CLASSES:

            source_class_dir = os.path.join(
                SOURCE_DIR,
                class_name
            )

            destination_class_dir = os.path.join(
                OUTPUT_DIR,
                split_name,
                class_name
            )

            for folder in match_class_sequences[match][class_name]:

                source = os.path.join(
                    source_class_dir,
                    folder
                )

                destination = os.path.join(
                    destination_class_dir,
                    folder
                )

                shutil.copytree(source, destination)

                total += 1

    return total


train_total = copy_split(train_matches, "train")
val_total = copy_split(val_matches, "val")
test_total = copy_split(test_matches, "test")


# --------------------------------------------------
# Report class distribution
# --------------------------------------------------

def get_split_counts(split_name):

    counts = Counter()

    for class_name in CLASSES:

        path = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        counts[class_name] = len([
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ])

    return counts


train_counts = get_split_counts("train")
val_counts = get_split_counts("val")
test_counts = get_split_counts("test")


print("\nCLASS DISTRIBUTION")
print("=" * 60)

print(f"{'CLASS':10} {'TRAIN':>8} {'VAL':>8} {'TEST':>8}")

for class_name in CLASSES:

    print(
        f"{class_name.upper():10} "
        f"{train_counts[class_name]:8} "
        f"{val_counts[class_name]:8} "
        f"{test_counts[class_name]:8}"
    )


# --------------------------------------------------
# Leakage check
# --------------------------------------------------

train_set = set(train_matches)
val_set = set(val_matches)
test_set = set(test_matches)

print("\nLEAKAGE CHECK")
print("=" * 60)

print("Train ∩ Validation :", len(train_set & val_set))
print("Train ∩ Test       :", len(train_set & test_set))
print("Validation ∩ Test  :", len(val_set & test_set))

if (
    len(train_set & val_set) == 0
    and len(train_set & test_set) == 0
    and len(val_set & test_set) == 0
):
    print("\nDataset split is SAFE.")
else:
    print("\nWARNING: MATCH LEAKAGE DETECTED!")


print("\n============================================================")
print("SPLIT COMPLETED")
print("============================================================")
print(f"Train sequences      : {train_total}")
print(f"Validation sequences : {val_total}")
print(f"Test sequences       : {test_total}")

print("\nCreated:", OUTPUT_DIR)