import os
import random
import shutil

SOURCE_DIR = "data/processed"
OUTPUT_DIR = "data/split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["smash", "clear", "drop", "drive", "net"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FRAMES_PER_SEQUENCE = 16
RANDOM_SEED = 42


def get_valid_sequences(class_dir):
    sequences = []

    if not os.path.exists(class_dir):
        return sequences

    for sequence_name in sorted(os.listdir(class_dir)):
        sequence_dir = os.path.join(class_dir, sequence_name)

        if not os.path.isdir(sequence_dir):
            continue

        frames = [
            file for file in os.listdir(sequence_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(frames) >= FRAMES_PER_SEQUENCE:
            sequences.append(sequence_name)

    return sequences


def split_sequences(sequences):
    sequences = sequences.copy()

    random.shuffle(sequences)

    total = len(sequences)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_sequences = sequences[:train_count]

    validation_sequences = sequences[
        train_count:train_count + val_count
    ]

    test_sequences = sequences[
        train_count + val_count:
    ]

    return train_sequences, validation_sequences, test_sequences


def copy_sequences(class_name, split_name, sequences):
    source_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    destination_dir = os.path.join(
        OUTPUT_DIR,
        split_name,
        class_name
    )

    os.makedirs(destination_dir, exist_ok=True)

    for sequence_name in sequences:

        source_sequence = os.path.join(
            source_dir,
            sequence_name
        )

        destination_sequence = os.path.join(
            destination_dir,
            sequence_name
        )

        shutil.copytree(
            source_sequence,
            destination_sequence
        )


def check_no_overlap(train, validation, test, class_name):

    train_set = set(train)
    validation_set = set(validation)
    test_set = set(test)

    train_validation = train_set & validation_set
    train_test = train_set & test_set
    validation_test = validation_set & test_set

    if train_validation:
        raise RuntimeError(
            f"LEAKAGE FOUND in {class_name}: "
            f"Train/Validation overlap = {train_validation}"
        )

    if train_test:
        raise RuntimeError(
            f"LEAKAGE FOUND in {class_name}: "
            f"Train/Test overlap = {train_test}"
        )

    if validation_test:
        raise RuntimeError(
            f"LEAKAGE FOUND in {class_name}: "
            f"Validation/Test overlap = {validation_test}"
        )


def main():

    random.seed(RANDOM_SEED)

    print("=" * 50)
    print("BADMINTON DATASET SPLITTING")
    print("=" * 50)

    print(f"Train      : {TRAIN_RATIO * 100:.0f}%")
    print(f"Validation : {VAL_RATIO * 100:.0f}%")
    print(f"Test       : {TEST_RATIO * 100:.0f}%")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    if abs(
        TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
    ) > 0.0001:

        raise ValueError(
            "Train + Validation + Test ratios must equal 1.0"
        )

    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(
            f"Source folder not found: {SOURCE_DIR}"
        )

    # Remove old split so we create a completely fresh fixed split
    if os.path.exists(OUTPUT_DIR):
        print("Removing previous split...")
        shutil.rmtree(OUTPUT_DIR)

    total_train = 0
    total_validation = 0
    total_test = 0

    all_train_sequences = set()
    all_validation_sequences = set()
    all_test_sequences = set()

    for class_name in CLASSES:

        class_dir = os.path.join(
            SOURCE_DIR,
            class_name
        )

        sequences = get_valid_sequences(class_dir)

        print(f"{class_name.upper()}")
        print(f"Valid sequences: {len(sequences)}")

        train, validation, test = split_sequences(sequences)

        # Check leakage before copying
        check_no_overlap(
            train,
            validation,
            test,
            class_name
        )

        copy_sequences(
            class_name,
            "train",
            train
        )

        copy_sequences(
            class_name,
            "validation",
            validation
        )

        copy_sequences(
            class_name,
            "test",
            test
        )

        total_train += len(train)
        total_validation += len(validation)
        total_test += len(test)

        all_train_sequences.update(
            f"{class_name}/{x}" for x in train
        )

        all_validation_sequences.update(
            f"{class_name}/{x}" for x in validation
        )

        all_test_sequences.update(
            f"{class_name}/{x}" for x in test
        )

        print(f"Train      : {len(train)}")
        print(f"Validation : {len(validation)}")
        print(f"Test       : {len(test)}")
        print()

    # Final global leakage check
    if all_train_sequences & all_validation_sequences:
        raise RuntimeError(
            "GLOBAL LEAKAGE FOUND: Train/Validation overlap!"
        )

    if all_train_sequences & all_test_sequences:
        raise RuntimeError(
            "GLOBAL LEAKAGE FOUND: Train/Test overlap!"
        )

    if all_validation_sequences & all_test_sequences:
        raise RuntimeError(
            "GLOBAL LEAKAGE FOUND: Validation/Test overlap!"
        )

    print("=" * 50)
    print("SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(f"Total Train      : {total_train}")
    print(f"Total Validation : {total_validation}")
    print(f"Total Test       : {total_test}")
    print(
        f"Total Sequences  : "
        f"{total_train + total_validation + total_test}"
    )

    print()
    print("LEAKAGE CHECK")
    print("=" * 50)
    print("Train ∩ Validation : 0")
    print("Train ∩ Test       : 0")
    print("Validation ∩ Test  : 0")
    print()
    print("Dataset split is SAFE.")
    print()
    print(f"Created: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
