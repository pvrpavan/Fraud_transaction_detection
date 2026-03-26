"""
Data augmentation script to create a more realistic fraud detection dataset.

Adds legitimate transactions that mimic fraud patterns (hard negatives),
making the classification problem more realistic and challenging.
In real-world banking:
  - People legitimately transfer their full balance (account closing, consolidation)
  - Large cash-outs happen regularly (payroll, large purchases)
  - Destination balances don't always update instantly (batch processing)
"""

import csv
import random
import math
import os

random.seed(42)

INPUT_FILE = "filtered_rows.csv"
OUTPUT_FILE = "data/augmented_transactions.csv"

# Columns to exclude from the output (identifiers that don't help prediction)
EXCLUDE_COLS = ["nameOrig", "nameDest"]


def read_data(path):
    """Read CSV and separate fraud/legit rows."""
    fraud, legit = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            if row["isFraud"] == "1":
                fraud.append(row)
            else:
                legit.append(row)
    # Remove excluded columns from headers
    headers = [h for h in headers if h not in EXCLUDE_COLS]
    return headers, fraud, legit


def generate_hard_negatives(fraud_rows, n_total=10000):
    """
    Generate legitimate transactions that resemble fraud patterns.

    These represent real-world scenarios where legitimate transactions
    share characteristics with fraudulent ones:
    1. Fraud-pattern clones (legitimate txns with perturbed fraud signatures)
    2. Full balance transfers (account consolidation, closing accounts)
    3. Full drain with delayed dest update (batch processing)
    4. Near-full drain (90-99% of balance transferred)
    5. Large amount transactions (similar magnitude to fraud)
    6. Noisy fraud-like patterns (blurred fraud signatures)
    7. Zero-balance origin transactions (new accounts)
    """
    hard_negatives = []

    # Get fraud amount distribution for realistic amounts
    fraud_amounts = [float(r["amount"]) for r in fraud_rows]
    fraud_amounts.sort()

    # Get fraud oldbalanceOrg distribution
    fraud_old_bal_org = [float(r["oldbalanceOrg"]) for r in fraud_rows]

    # Get fraud oldbalanceDest distribution
    fraud_old_bal_dest = [float(r["oldbalanceDest"]) for r in fraud_rows]

    # Distribution of steps (time) from fraud
    fraud_steps = [int(r["step"]) for r in fraud_rows]

    # --- Type 1: Fraud-pattern clones (perturbed copies of fraud patterns) ---
    # These are legitimate transactions that resemble fraud signatures
    # but with large perturbation (30-60%). In real banking, some
    # legit transactions share characteristics with fraud.
    n_type1 = int(n_total * 0.10)
    for _ in range(n_type1):
        # Pick a random fraud row and clone its pattern with large perturbation
        template = random.choice(fraud_rows)
        amount = float(template["amount"])
        old_bal_org = float(template["oldbalanceOrg"])
        old_bal_dest = float(template["oldbalanceDest"])
        step = int(template["step"])
        txn_type = template["type"]

        # Apply large perturbation (30-60%) so strong models can learn the difference
        perturbation = random.uniform(0.40, 1.60)
        amount = round(amount * perturbation, 2)
        if old_bal_org > 0:
            old_bal_org = round(old_bal_org * random.uniform(0.50, 1.50), 2)
        if old_bal_dest > 0:
            old_bal_dest = round(old_bal_dest * random.uniform(0.30, 2.00), 2)
        new_bal_orig = round(max(0, old_bal_org - amount), 2)
        # Dest usually updates correctly for legit transactions
        if random.random() < 0.8:
            new_bal_dest = round(old_bal_dest + amount, 2)
        else:
            new_bal_dest = round(old_bal_dest + amount * random.uniform(0.5, 1.0), 2)
        step = max(1, step + random.randint(-30, 30))

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    # --- Type 2: Full balance drain transfers (legitimate account consolidation) ---
    n_type2 = int(n_total * 0.22)
    for _ in range(n_type2):
        amount = random.choice(fraud_amounts) * random.uniform(0.3, 1.5)
        amount = round(amount, 2)
        old_bal_org = amount  # Full drain
        new_bal_orig = 0.0
        old_bal_dest = round(random.choice(fraud_old_bal_dest) * random.uniform(0.5, 2.0), 2)
        new_bal_dest = round(old_bal_dest + amount, 2)
        step = random.choice(fraud_steps) + random.randint(-10, 10)
        step = max(1, step)
        txn_type = random.choice(["TRANSFER", "CASH_OUT"])

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    # --- Type 3: Full drain with delayed dest update (batch processing) ---
    n_type3 = int(n_total * 0.18)
    for _ in range(n_type3):
        amount = random.choice(fraud_amounts) * random.uniform(0.5, 1.5)
        amount = round(amount, 2)
        old_bal_org = amount
        new_bal_orig = 0.0
        old_bal_dest = round(random.choice(fraud_old_bal_dest) * random.uniform(0.5, 1.5), 2)
        new_bal_dest = old_bal_dest  # Dest doesn't change (batch processing)
        step = random.choice(fraud_steps) + random.randint(-3, 3)
        step = max(1, step)
        txn_type = random.choice(["TRANSFER", "CASH_OUT"])

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    # --- Type 4: Near-full drain (90-100% of balance transferred) ---
    n_type4 = int(n_total * 0.15)
    for _ in range(n_type4):
        old_bal_org = random.choice(fraud_old_bal_org) * random.uniform(0.5, 2.0)
        old_bal_org = round(max(old_bal_org, 100), 2)
        drain_pct = random.uniform(0.90, 1.00)
        amount = round(old_bal_org * drain_pct, 2)
        new_bal_orig = round(old_bal_org - amount, 2)
        old_bal_dest = round(random.choice(fraud_old_bal_dest) * random.uniform(0.3, 2.0), 2)
        # Mix of correct, partial, and no dest updates
        r = random.random()
        if r < 0.33:
            new_bal_dest = round(old_bal_dest + amount, 2)
        elif r < 0.66:
            new_bal_dest = round(old_bal_dest + amount * random.uniform(0, 0.3), 2)
        else:
            new_bal_dest = old_bal_dest
        step = random.choice(fraud_steps) + random.randint(-10, 10)
        step = max(1, step)
        txn_type = random.choice(["TRANSFER", "CASH_OUT"])

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    # --- Type 5: Large amount transactions (similar magnitude to fraud) ---
    n_type5 = int(n_total * 0.15)
    for _ in range(n_type5):
        amount = random.choice(fraud_amounts) * random.uniform(0.8, 1.5)
        amount = round(amount, 2)
        old_bal_org = round(amount * random.uniform(1.0, 1.5), 2)
        new_bal_orig = round(old_bal_org - amount, 2)
        old_bal_dest = round(random.choice(fraud_old_bal_dest) * random.uniform(0.5, 2.0), 2)
        new_bal_dest = round(old_bal_dest + amount, 2)
        step = random.choice(fraud_steps) + random.randint(-20, 20)
        step = max(1, step)
        txn_type = random.choice(["TRANSFER", "CASH_OUT"])

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    # --- Type 6: Noisy fraud-like patterns (blurred fraud signatures) ---
    # These add noise within the fraud feature space to blur decision boundaries
    n_type6 = int(n_total * 0.12)
    for _ in range(n_type6):
        template = random.choice(fraud_rows)
        amount = float(template["amount"]) * random.uniform(0.5, 1.5)
        amount = round(amount, 2)
        old_bal_org = round(amount * random.uniform(0.80, 1.30), 2)
        new_bal_orig = round(max(0, old_bal_org - amount), 2)
        old_bal_dest = float(template["oldbalanceDest"]) * random.uniform(0.3, 2.5)
        old_bal_dest = round(max(0, old_bal_dest), 2)
        new_bal_dest = round(old_bal_dest + amount * random.uniform(0.3, 1.0), 2)
        step = int(template["step"]) + random.randint(-30, 30)
        step = max(1, step)
        txn_type = template["type"]

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    # --- Type 7: Zero-balance origin transactions (new accounts) ---
    n_type7 = n_total - n_type1 - n_type2 - n_type3 - n_type4 - n_type5 - n_type6
    for _ in range(n_type7):
        amount = random.choice(fraud_amounts) * random.uniform(0.5, 1.5)
        amount = round(amount, 2)
        old_bal_org = 0.0
        new_bal_orig = 0.0
        old_bal_dest = round(random.choice(fraud_old_bal_dest) * random.uniform(0.3, 2.0), 2)
        new_bal_dest = round(old_bal_dest + amount * random.uniform(0.5, 1.0), 2)
        step = random.choice(fraud_steps) + random.randint(-10, 10)
        step = max(1, step)
        txn_type = random.choice(["TRANSFER", "CASH_OUT"])

        hard_negatives.append({
            "step": str(step),
            "type": txn_type,
            "amount": f"{amount:.2f}",
            "nameOrig": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceOrg": f"{old_bal_org:.2f}",
            "newbalanceOrig": f"{new_bal_orig:.2f}",
            "nameDest": f"C{random.randint(100000000, 999999999)}",
            "oldbalanceDest": f"{old_bal_dest:.2f}",
            "newbalanceDest": f"{new_bal_dest:.2f}",
            "isFraud": "0",
            "isFlaggedFraud": "0",
        })

    return hard_negatives


def main():
    print("Reading original data...")
    headers, fraud, legit = read_data(INPUT_FILE)

    print(f"Original: {len(fraud)} fraud + {len(legit)} legit = {len(fraud)+len(legit)} total")
    print(f"Original fraud ratio: {len(fraud)/(len(fraud)+len(legit)):.4%}")

    print("\nGenerating hard negatives...")
    hard_negatives = generate_hard_negatives(fraud, n_total=10000)
    print(f"Generated {len(hard_negatives)} hard negative transactions")

    # Remove excluded columns from all rows
    for row_list in [fraud, legit, hard_negatives]:
        for row in row_list:
            for col in EXCLUDE_COLS:
                row.pop(col, None)

    # Combine all data
    all_rows = fraud + legit + hard_negatives
    random.shuffle(all_rows)

    total = len(all_rows)
    fraud_count = sum(1 for r in all_rows if r["isFraud"] == "1")
    print(f"\nAugmented: {fraud_count} fraud + {total-fraud_count} legit = {total} total")
    print(f"Augmented fraud ratio: {fraud_count/total:.4%}")

    # Write augmented dataset
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nAugmented dataset saved to {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
