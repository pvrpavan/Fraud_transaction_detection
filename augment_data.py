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
    return headers, fraud, legit


def generate_hard_negatives(fraud_rows, n_total=60000):
    """
    Generate legitimate transactions that resemble fraud patterns.

    These represent real-world scenarios where legitimate transactions
    share characteristics with fraudulent ones:
    1. Full balance transfers (account consolidation, closing accounts)
    2. Large cash-outs (payroll, business expenses)
    3. Transactions where destination balance doesn't update instantly
    """
    hard_negatives = []

    # Get fraud amount distribution for realistic amounts
    fraud_amounts = [float(r["amount"]) for r in fraud_rows]
    fraud_amounts.sort()

    # Distribution of steps (time) from fraud
    fraud_steps = [int(r["step"]) for r in fraud_rows]

    # --- Type 1: Full balance drain transfers (legitimate account consolidation) ---
    # ~8000 rows: amount == oldbalanceOrg, newbalanceOrig == 0, dest updates correctly
    n_type1 = int(n_total * 0.32)
    for _ in range(n_type1):
        amount = random.choice(fraud_amounts) * random.uniform(0.3, 1.5)
        amount = round(amount, 2)
        old_bal_org = amount  # Full drain
        new_bal_orig = 0.0
        old_bal_dest = round(random.uniform(0, 5000000), 2)
        new_bal_dest = round(old_bal_dest + amount, 2)  # Dest balance updates correctly
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

    # --- Type 2: Full drain with delayed dest update (batch processing) ---
    # ~5000 rows: amount == oldbalanceOrg, dest balance unchanged (looks like fraud)
    n_type2 = int(n_total * 0.20)
    for _ in range(n_type2):
        amount = random.choice(fraud_amounts) * random.uniform(0.5, 1.2)
        amount = round(amount, 2)
        old_bal_org = amount
        new_bal_orig = 0.0
        old_bal_dest = round(random.uniform(0, 3000000), 2)
        new_bal_dest = old_bal_dest  # Dest doesn't change (batch processing)
        step = random.choice(fraud_steps) + random.randint(-5, 5)
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

    # --- Type 3: Near-full drain (90-99% of balance transferred) ---
    # ~5000 rows: amount is close to oldbalanceOrg but not exact
    n_type3 = int(n_total * 0.20)
    for _ in range(n_type3):
        old_bal_org = random.choice(fraud_amounts) * random.uniform(0.5, 2.0)
        old_bal_org = round(old_bal_org, 2)
        drain_pct = random.uniform(0.90, 0.99)
        amount = round(old_bal_org * drain_pct, 2)
        new_bal_orig = round(old_bal_org - amount, 2)
        old_bal_dest = round(random.uniform(0, 4000000), 2)
        # Mix of correct and delayed dest updates
        if random.random() < 0.5:
            new_bal_dest = round(old_bal_dest + amount, 2)
        else:
            new_bal_dest = round(old_bal_dest + amount * random.uniform(0, 0.3), 2)
        step = random.choice(fraud_steps) + random.randint(-15, 15)
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

    # --- Type 4: Large amount transactions (similar magnitude to fraud) ---
    # ~4000 rows: large amounts but partial balance usage
    n_type4 = int(n_total * 0.16)
    for _ in range(n_type4):
        amount = random.choice(fraud_amounts) * random.uniform(0.8, 1.5)
        amount = round(amount, 2)
        old_bal_org = round(amount * random.uniform(1.1, 3.0), 2)
        new_bal_orig = round(old_bal_org - amount, 2)
        old_bal_dest = round(random.uniform(0, 5000000), 2)
        new_bal_dest = round(old_bal_dest + amount, 2)
        step = random.randint(1, 743)
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

    # --- Type 5: Zero-balance origin transactions (new accounts making transfers) ---
    # ~3000 rows: oldbalanceOrg == 0 (deposited and immediately transferred)
    n_type5 = n_total - n_type1 - n_type2 - n_type3 - n_type4
    for _ in range(n_type5):
        amount = round(random.uniform(1000, 500000), 2)
        old_bal_org = 0.0
        new_bal_orig = 0.0
        old_bal_dest = round(random.uniform(0, 2000000), 2)
        new_bal_dest = round(old_bal_dest + amount * random.uniform(0.5, 1.0), 2)
        step = random.randint(1, 743)
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
    hard_negatives = generate_hard_negatives(fraud, n_total=60000)
    print(f"Generated {len(hard_negatives)} hard negative transactions")

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
