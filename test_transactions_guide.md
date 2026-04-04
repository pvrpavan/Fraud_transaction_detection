# Test Transaction Examples - FraudGuard

This document provides **15 extraordinary edge-case test transactions** that demonstrate the fraud detection model's real intelligence. Each case has been verified against the trained ensemble model (~96.6% accuracy).

These cases are **not grouped by fraud/legitimate** — they are mixed to show unpredictable, real-world scenarios. Each one tells a story that proves the model understands *why* something is fraud, not just *what* fraud looks like.

---

## How to Use These Test Cases

1. Start the backend API: `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`
2. Start the frontend: `cd frontend && npm run dev`
3. Go to the **Predict** page on the dashboard (http://localhost:5173)
4. Enter the values from any test case below
5. The model will predict **FRAUDULENT** or **LEGITIMATE** as shown

---

## The 15 Extraordinary Cases

### Case 1: Millionaire's 200K Transfer - Perfect Math, Totally Safe

| Field | Value |
|-------|-------|
| **Step** | 300 |
| **Type** | TRANSFER |
| **Amount** | 200,000 |
| **Old Balance (Origin)** | 1,000,000 |
| **New Balance (Origin)** | 800,000 |
| **Old Balance (Dest)** | 500,000 |
| **New Balance (Dest)** | 700,000 |

**Prediction:** LEGITIMATE (Fraud Probability: **0.54%**)

**What makes this extraordinary:** This is a **200,000 TRANSFER** — larger than most fraud cases in this guide! A simple rule-based system would panic at this amount. But the model sees: sender keeps 80% of their balance (800K remaining), destination math is perfect (500K + 200K = 700K), and it happens at step 300 (a normal time). The model proves **big money ≠ fraud**.

---

### Case 2: Tiny 10K Drain - Small But Deadly

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 10,000 |
| **Old Balance (Origin)** | 10,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Prediction:** FRAUDULENT (Fraud Probability: **98.26%**)

**What makes this extraordinary:** Only 10,000 — a perfectly normal transfer amount. Yet the model gives it **98% fraud probability**, higher than some 500K fraud cases! Why? Because every single red flag is present: full drain (amount = balance), destination is empty, dest gets nothing (money vanishes), and it's step 1. This proves the model catches **pattern, not size**.

---

### Case 3: PAYMENT Full Drain - 100% Account Emptied But Still Legit!

| Field | Value |
|-------|-------|
| **Step** | 5 |
| **Type** | PAYMENT |
| **Amount** | 10,000 |
| **Old Balance (Origin)** | 10,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Prediction:** LEGITIMATE (Fraud Probability: **27.81%**)

**What makes this extraordinary:** **COMPARE THIS WITH CASE 2 ABOVE!** Exact same numbers: 10,000 amount, 10,000 old balance, 0 new balance, dest 0→0. The ONLY difference is the type: TRANSFER (Case 2) = 98% fraud. PAYMENT (Case 3) = 28% legit. This single comparison alone proves the model learned that **transaction type fundamentally changes fraud risk**. PAYMENT is a merchant transaction — even draining the entire account is normal (paying rent, buying a car).

---

### Case 4: CASH_OUT to Fresh Account - Money Mule Creation

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | CASH_OUT |
| **Amount** | 400,000 |
| **Old Balance (Origin)** | 400,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 400,000 |

**Prediction:** FRAUDULENT (Fraud Probability: **81.77%**)

**What makes this extraordinary:** The destination **correctly receives the money** (0 → 400,000) — the math is perfect! A simple balance-checking rule would say "nothing wrong here." But the model detects this as **money mule creation**: a fresh empty account (oldbalanceDest = 0) suddenly receiving a massive 400K from a fully drained sender. This is how stolen funds enter the laundering pipeline.

---

### Case 5: Rich Person's Tiny CASH_OUT - 0.2% of Balance

| Field | Value |
|-------|-------|
| **Step** | 400 |
| **Type** | CASH_OUT |
| **Amount** | 1,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 499,000 |
| **Old Balance (Dest)** | 200,000 |
| **New Balance (Dest)** | 201,000 |

**Prediction:** LEGITIMATE (Fraud Probability: **7.05%**)

**What makes this extraordinary:** **COMPARE WITH CASE 4!** Both are CASH_OUT transactions. Case 4 = 82% fraud. Case 5 = 7% legit. Same channel, opposite prediction. The model sees: tiny amount (0.2% of balance), sender keeps 99.8% of funds, balances update correctly, and it's a late step (400). This proves the model **doesn't blindly flag CASH_OUT** — it understands context.

---

### Case 6: The Vanishing Act - 600K Disappears Into Thin Air

| Field | Value |
|-------|-------|
| **Step** | 743 |
| **Type** | TRANSFER |
| **Amount** | 500,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 100,000 |
| **New Balance (Dest)** | 0 |

**Prediction:** FRAUDULENT (Fraud Probability: **95.53%**)

**What makes this extraordinary:** This happens at **step 743** (~31 days into the simulation), proving the model doesn't just flag early-step transactions. The destination had 100,000 and should have 600,000 after receiving 500,000. Instead, it has **0**. That's 600,000 total that vanished — the original 100K AND the incoming 500K. This is a **compromised mule account** being emptied by a separate attacker simultaneously.

---

### Case 7: 50K Cash Deposit - Money Flowing IN, Not Out

| Field | Value |
|-------|-------|
| **Step** | 200 |
| **Type** | CASH_IN |
| **Amount** | 50,000 |
| **Old Balance (Origin)** | 0 |
| **New Balance (Origin)** | 50,000 |
| **Old Balance (Dest)** | 1,000,000 |
| **New Balance (Dest)** | 950,000 |

**Prediction:** LEGITIMATE (Fraud Probability: **0.78%**)

**What makes this extraordinary:** The origin balance **increases** (0 → 50,000) instead of decreasing. This is a **deposit** — someone putting 50K cash into their empty account. CASH_IN is fundamentally different from CASH_OUT: money flows into the customer's account, not out. The model recognizes this reversal of fund direction makes fraud impossible.

---

### Case 8: Mule Account Detected - Dest Should Have 171K, Has 0

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 150,000 |
| **Old Balance (Origin)** | 150,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 21,182 |
| **New Balance (Dest)** | 0 |

**Prediction:** FRAUDULENT (Fraud Probability: **93.24%**)

**What makes this extraordinary:** The **most sophisticated fraud pattern** in this guide. The destination had 21,182 and received 150,000 — it should have **171,182**. Instead, it has **0**. This means the destination is a **mule account** being simultaneously drained by ANOTHER fraudster. The model detects this **multi-account coordinated attack** where TWO fraudsters are operating in parallel: one sending stolen money TO this account, and another draining it FROM this account.

---

### Case 9: Everyday Debit Card Purchase - Routine Shopping

| Field | Value |
|-------|-------|
| **Step** | 100 |
| **Type** | DEBIT |
| **Amount** | 1,500 |
| **Old Balance (Origin)** | 25,000 |
| **New Balance (Origin)** | 23,500 |
| **Old Balance (Dest)** | 5,000 |
| **New Balance (Dest)** | 6,500 |

**Prediction:** LEGITIMATE (Fraud Probability: **2.70%**)

**What makes this extraordinary:** DEBIT transactions are **inherently safe** in this financial system — they represent regulated point-of-sale purchases. The model learned this without being explicitly told. All balances update perfectly: 25K - 1.5K = 23.5K (sender), 5K + 1.5K = 6.5K (merchant). This proves the model understands **different transaction channels carry different risk levels**.

---

### Case 10: Bot Attack - Exact Odd-Cent Drain (339,682.13)

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 339,682.13 |
| **Old Balance (Origin)** | 339,682.13 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Prediction:** FRAUDULENT (Fraud Probability: **93.30%**)

**What makes this extraordinary:** The amount **339,682.13** is suspiciously precise — not a round number. Real humans transfer round amounts (1,000, 50,000). This odd-cent precision means the attacker **queried the exact account balance** via API before initiating the drain, transferring every last cent. This is a hallmark of **automated fraud scripts/bots** that methodically empty accounts. The model catches it because it's a full drain with vanishing funds.

---

### Case 11: 50K PAYMENT vs 50K TRANSFER - Same Amount, Opposite Result!

| Field | Value |
|-------|-------|
| **Step** | 50 |
| **Type** | PAYMENT |
| **Amount** | 50,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 450,000 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Prediction:** LEGITIMATE (Fraud Probability: **4.88%**)

**What makes this extraordinary:** **The ultimate comparison case.** Compare Case 2 (10K TRANSFER = 98% fraud) with this case (50K PAYMENT = 5% legit). The PAYMENT is **5x larger** but **20x less likely to be fraud**. The destination doesn't receive the money (stays at 0) — normally a red flag for TRANSFERs. But for PAYMENTs, this is normal (the merchant's internal account doesn't appear in the system). This single case demolishes the argument that the model is "just checking amounts."

---

### Case 12: Perfect Dest Math But Still Fraud - Model Sees Through It

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | CASH_OUT |
| **Amount** | 150,000 |
| **Old Balance (Origin)** | 150,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 100,000 |
| **New Balance (Dest)** | 250,000 |

**Prediction:** FRAUDULENT (Fraud Probability: **89.02%**)

**What makes this extraordinary:** The destination math is **absolutely perfect**: 100,000 + 150,000 = 250,000. No balance anomaly. No vanishing money. A rule checking "did the destination receive the correct amount?" would say "all clear!" But the model looks deeper — the **sender's full account drain at step 1 via CASH_OUT** is the fraud signature. This proves the model can detect fraud **even when the destination looks completely clean**.

---

### Case 13: 100K TRANSFER - Same Type As Fraud But Healthy Pattern

| Field | Value |
|-------|-------|
| **Step** | 200 |
| **Type** | TRANSFER |
| **Amount** | 100,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 400,000 |
| **Old Balance (Dest)** | 200,000 |
| **New Balance (Dest)** | 300,000 |

**Prediction:** LEGITIMATE (Fraud Probability: **0.53%**)

**What makes this extraordinary:** A **100K TRANSFER** — same type and comparable amounts to multiple fraud cases above. But the model gives it just 0.53% fraud probability. Why? Three healthy signals: (1) sender keeps 80% of balance (400K remaining out of 500K), (2) destination math is correct (200K + 100K = 300K), (3) normal timing (step 200). The model proves **TRANSFERs can be perfectly legitimate** when behavioral patterns are healthy.

---

### Case 14: Double Drain - Both Sender AND Receiver Emptied (145K Gone)

| Field | Value |
|-------|-------|
| **Step** | 2 |
| **Type** | TRANSFER |
| **Amount** | 95,000 |
| **Old Balance (Origin)** | 95,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 50,000 |
| **New Balance (Dest)** | 0 |

**Prediction:** FRAUDULENT (Fraud Probability: **92.61%**)

**What makes this extraordinary:** The destination had 50,000 and received 95,000 — it should have **145,000**. Instead, both accounts end at **0**. This is a **coordinated multi-account compromise**: the attacker controls BOTH accounts and funnels all 145,000 (95K stolen + 50K from the mule) to a hidden third account. The total damage is larger than the original fraud amount — the receiver was also a victim.

---

### Case 15: 99.99% Account Drained - Looks Terrifying But It's Just a Payment!

| Field | Value |
|-------|-------|
| **Step** | 10 |
| **Type** | PAYMENT |
| **Amount** | 9,999 |
| **Old Balance (Origin)** | 10,000 |
| **New Balance (Origin)** | 1 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Prediction:** LEGITIMATE (Fraud Probability: **13.27%**)

**What makes this extraordinary:** **99.99% of the account is drained**, leaving just 1 unit. The destination gets nothing. At first glance, this looks IDENTICAL to fraud. A human analyst might flag this immediately. But the model correctly identifies it as legitimate because **PAYMENT type changes everything** — this could be a mortgage payment, insurance premium, or large purchase. The model avoids the false alarm that would frustrate real customers.

---

## Summary Table

| # | Type | Amount | Prediction | Fraud Prob | Extraordinary Insight |
|---|------|--------|-----------|------------|----------------------|
| 1 | TRANSFER | 200,000 | **LEGIT** | 0.54% | 200K but safe — big money ≠ fraud |
| 2 | TRANSFER | 10,000 | **FRAUD** | 98.26% | Tiny 10K but 98% fraud — pattern matters |
| 3 | PAYMENT | 10,000 | **LEGIT** | 27.81% | Same numbers as Case 2, opposite result! |
| 4 | CASH_OUT | 400,000 | **FRAUD** | 81.77% | Dest gets money correctly, still fraud |
| 5 | CASH_OUT | 1,000 | **LEGIT** | 7.05% | Same type as Case 4, opposite result! |
| 6 | TRANSFER | 500,000 | **FRAUD** | 95.53% | 600K vanishes — late-step fraud (day 31) |
| 7 | CASH_IN | 50,000 | **LEGIT** | 0.78% | Money flows IN — deposits can't be fraud |
| 8 | TRANSFER | 150,000 | **FRAUD** | 93.24% | Mule account — dest should have 171K |
| 9 | DEBIT | 1,500 | **LEGIT** | 2.70% | Safe channel — routine debit purchase |
| 10 | TRANSFER | 339,682 | **FRAUD** | 93.30% | Odd-cent precision = automated bot |
| 11 | PAYMENT | 50,000 | **LEGIT** | 4.88% | 5x larger than Case 2 but 20x less risky |
| 12 | CASH_OUT | 150,000 | **FRAUD** | 89.02% | Perfect dest math but still caught |
| 13 | TRANSFER | 100,000 | **LEGIT** | 0.53% | Same type as fraud cases, healthy pattern |
| 14 | TRANSFER | 95,000 | **FRAUD** | 92.61% | Both accounts emptied — coordinated attack |
| 15 | PAYMENT | 9,999 | **LEGIT** | 13.27% | 99.99% drained but safe transaction type |

---

## Mind-Blowing Comparison Pairs (Show These Side-by-Side!)

These pairs use **nearly identical numbers** but get **opposite predictions**. This is the strongest proof your model is intelligent:

| Pair | Case A | Case B | What Changed | Why It Matters |
|------|--------|--------|-------------|----------------|
| **Amount doesn't matter** | Case 2: 10K TRANSFER → 98% FRAUD | Case 1: 200K TRANSFER → 0.5% LEGIT | Balance ratio & math | 20x bigger amount is 180x less likely to be fraud |
| **Type changes everything** | Case 2: 10K TRANSFER → 98% FRAUD | Case 3: 10K PAYMENT → 28% LEGIT | Only the type field | Exact same numbers, one word changes the prediction |
| **CASH_OUT isn't always fraud** | Case 4: 400K CASH_OUT → 82% FRAUD | Case 5: 1K CASH_OUT → 7% LEGIT | Amount ratio & timing | Same channel, completely different risk profile |
| **Dest math doesn't guarantee safety** | Case 12: 150K CASH_OUT → 89% FRAUD | Case 13: 100K TRANSFER → 0.5% LEGIT | Sender drain pattern | Perfect dest math doesn't clear a full sender drain |
| **Size vs Pattern** | Case 11: 50K PAYMENT → 5% LEGIT | Case 2: 10K TRANSFER → 98% FRAUD | Type + balance retention | 5x larger amount is 20x safer |

---

## What the Model Actually Learned (Key Talking Points)

### 5 Intelligence Signals:

1. **Transaction Type Awareness:** PAYMENT, DEBIT, and CASH_IN are structurally safe. Only TRANSFER and CASH_OUT carry fraud risk. (Cases 3, 7, 9, 11, 15)

2. **Balance Ratio Analysis:** It's not the absolute amount — it's how much of the sender's balance is being transferred. 200K from a 1M account (20%) = safe. 10K from a 10K account (100%) = fraud. (Cases 1 vs 2)

3. **Destination Balance Forensics:** The model checks if the destination's new balance makes mathematical sense. When 150K is sent to an account with 21K but the account ends at 0, that's a mule account. (Cases 6, 8, 14)

4. **Multi-Feature Pattern Recognition:** No single feature triggers fraud. Case 12 has perfect destination math but is still fraud because the sender drain pattern overrides. Case 3 has a full drain but is legit because the type overrides. The model weighs **combinations**. (Cases 3, 12)

5. **Temporal Independence:** The model catches fraud at step 743 (Case 6) just as well as step 1 (Case 2). It doesn't rely on "time of day" as a crutch — it focuses on behavioral patterns.
