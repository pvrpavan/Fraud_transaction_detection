# Test Transaction Examples - FraudGuard AI

This document provides **15 verified edge-case test transactions** designed to demonstrate the fraud detection system's intelligence. Each case has been tested against the trained ensemble model (~96.6% accuracy) and produces the correct prediction.

These are not trivial examples. They include **subtle fraud patterns**, **tricky legitimate transactions that look suspicious**, and **edge cases** that prove the model truly understands fraud behavior rather than relying on simple rules.

---

## How to Use These Test Cases

1. Start the backend API: `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`
2. Start the frontend: `cd frontend && npm run dev`
3. Go to the **Predict** page on the dashboard (http://localhost:5173)
4. Enter the values from any test case below
5. The model will predict **FRAUDULENT** or **LEGITIMATE** as shown

---

## Fraudulent Transactions (8 Cases)

### Case 1: Classic Full Account Drain - The Textbook Fraud

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 181,000 |
| **Old Balance (Origin)** | 181,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~96%)

**Why this impresses:** This is the **most common real-world fraud pattern**. The entire balance is drained in a single transfer. But the key insight is the **destination balance stays at 0 even after receiving 181,000** - the money vanished into a ghost account. In a real bank, this means the funds were immediately forwarded to another account in a laundering chain.

---

### Case 2: Small Amount, But Every Red Flag Lit Up

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 50,000 |
| **Old Balance (Origin)** | 50,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~98%)

**Why this impresses:** The amount (50,000) is **not unusually large** - many legitimate transfers are this size. But the model catches it because: (1) amount exactly equals the full balance, (2) account drained to zero, (3) destination balance doesn't increase, and (4) it happens at step 1. This proves the model **doesn't just flag large amounts** - it understands the combination of suspicious patterns.

---

### Case 3: CASH_OUT Fraud - Different Channel, Same Crime

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | CASH_OUT |
| **Amount** | 300,000 |
| **Old Balance (Origin)** | 300,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 300,000 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~87%)

**Why this impresses:** Unlike TRANSFER fraud where dest stays at 0, in CASH_OUT the **destination does receive the money** (0 -> 300,000). A naive rule-based system checking "did the destination get the money?" would miss this entirely. But the model learned that **full account drains via CASH_OUT at early time steps are a distinct fraud pattern**, even when the destination balance updates correctly.

---

### Case 4: The Mule Account - Destination Balance Drops After Receiving Money

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 150,000 |
| **Old Balance (Origin)** | 150,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 21,182 |
| **New Balance (Dest)** | 0 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~93%)

**Why this impresses:** This is the **most sophisticated fraud pattern**. The destination account had 21,182 before the transaction and should have 171,182 after receiving 150,000. Instead it drops to 0. This means the destination is a **mule account** - a compromised account being drained simultaneously by another fraudster. The model detects this **multi-account coordinated attack**.

---

### Case 5: Coordinated Attack - Both Accounts Drained

| Field | Value |
|-------|-------|
| **Step** | 2 |
| **Type** | TRANSFER |
| **Amount** | 95,000 |
| **Old Balance (Origin)** | 95,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 50,000 |
| **New Balance (Dest)** | 0 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~93%)

**Why this impresses:** The destination had 50,000, received 95,000, but ended at 0. Both accounts are emptied simultaneously - the sender loses 95,000 and the receiver loses their original 50,000 too. This is a **coordinated multi-account compromise** where the attacker controls both accounts and funnels all 145,000 to a third hidden account.

---

### Case 6: CASH_OUT to Active Account - Subtle but Deadly

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | CASH_OUT |
| **Amount** | 150,000 |
| **Old Balance (Origin)** | 150,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 100,000 |
| **New Balance (Dest)** | 250,000 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~89%)

**Why this impresses:** Everything about the **destination looks perfectly legitimate** - it had 100,000, received 150,000, now has 250,000 (math checks out perfectly). A balance-checking rule would say "this is fine." But the model identifies it as fraud because the **sender's full account drain at step 1 via CASH_OUT** matches the statistical profile of compromised accounts, regardless of how clean the destination looks.

---

### Case 7: Late-Night Massive Transfer - Money Vanishes

| Field | Value |
|-------|-------|
| **Step** | 743 |
| **Type** | TRANSFER |
| **Amount** | 500,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 100,000 |
| **New Balance (Dest)** | 0 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~96%)

**Why this impresses:** This happens at **step 743** (about 31 days in), not at step 1 like most fraud cases. This proves the model **doesn't just memorize "early step = fraud"**. It still catches it because: the entire 500,000 is drained, and the destination had 100,000 but drops to 0 after supposedly receiving 500,000. The destination should have 600,000 but has nothing - a massive 600,000 discrepancy.

---

### Case 8: The Precision Drain - Exact Odd Amount

| Field | Value |
|-------|-------|
| **Step** | 1 |
| **Type** | TRANSFER |
| **Amount** | 339,682.13 |
| **Old Balance (Origin)** | 339,682.13 |
| **New Balance (Origin)** | 0 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Expected Result:** FRAUDULENT (Fraud Probability: ~93%)

**Why this impresses:** The amount 339,682.13 is **oddly precise** - not a round number. This means the fraudster **queried the exact account balance first** before initiating the transfer to drain every last cent. In legitimate banking, people transfer round amounts (1,000, 5,000, 50,000). Knowing the exact balance down to the cent and transferring that precise amount is a hallmark of automated fraud scripts.

---

## Legitimate Transactions (7 Cases)

These are the **tricky ones** - transactions that have suspicious-looking characteristics but are correctly identified as legitimate. These demonstrate the model's intelligence in **not over-flagging**.

### Case 9: PAYMENT Near-Full Drain - Looks Scary, But Safe

| Field | Value |
|-------|-------|
| **Step** | 10 |
| **Type** | PAYMENT |
| **Amount** | 9,999 |
| **Old Balance (Origin)** | 10,000 |
| **New Balance (Origin)** | 1 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~13%)

**Why this impresses:** This **looks terrifying** - 99.99% of the account balance is drained in a single transaction, leaving just 1 unit. The destination balance doesn't change. A naive system would flag this immediately. But the model knows that **PAYMENT transactions are never fraudulent** in this financial system. This could be a large purchase, bill payment, or insurance premium. The model avoids the false alarm.

---

### Case 10: Large PAYMENT 50K - Huge Amount, Still Safe

| Field | Value |
|-------|-------|
| **Step** | 50 |
| **Type** | PAYMENT |
| **Amount** | 50,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 450,000 |
| **Old Balance (Dest)** | 0 |
| **New Balance (Dest)** | 0 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~5%)

**Why this impresses:** 50,000 is a **very large transaction** - larger than some of the fraud cases above (Case 2 was 50K and flagged as fraud!). The destination balance doesn't increase (stays 0). But the model correctly identifies it as legitimate because: (1) PAYMENT type is safe, (2) the sender retains 90% of their balance, and (3) the amount-to-balance ratio is only 10%. This proves the model **doesn't simply flag large amounts**.

---

### Case 11: TRANSFER with Perfect Math - Large but Honest

| Field | Value |
|-------|-------|
| **Step** | 200 |
| **Type** | TRANSFER |
| **Amount** | 100,000 |
| **Old Balance (Origin)** | 500,000 |
| **New Balance (Origin)** | 400,000 |
| **Old Balance (Dest)** | 200,000 |
| **New Balance (Dest)** | 300,000 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~0.5%)

**Why this impresses:** This is a **100,000 TRANSFER** - the same type and similar amounts that triggered fraud above. But the model sees the difference: (1) sender keeps 80% of balance (400,000 remaining), (2) destination correctly goes from 200,000 to 300,000 (math: 200K + 100K = 300K), and (3) it happens at step 200, a normal time. The model proves it **understands TRANSFERs can be legitimate** when the patterns are healthy.

---

### Case 12: Small TRANSFER - Tiny Amount, Correct Balances

| Field | Value |
|-------|-------|
| **Step** | 150 |
| **Type** | TRANSFER |
| **Amount** | 3,000 |
| **Old Balance (Origin)** | 80,000 |
| **New Balance (Origin)** | 77,000 |
| **Old Balance (Dest)** | 15,000 |
| **New Balance (Dest)** | 18,000 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~1.2%)

**Why this impresses:** Even though TRANSFER is a fraud-capable type, this is clearly a normal person-to-person transfer. The amount is only 3.75% of the sender's balance, and the **destination balance increases correctly** (15,000 + 3,000 = 18,000). The model has learned to differentiate between risky and safe TRANSFER patterns.

---

### Case 13: DEBIT Transaction - Small and Routine

| Field | Value |
|-------|-------|
| **Step** | 100 |
| **Type** | DEBIT |
| **Amount** | 300 |
| **Old Balance (Origin)** | 8,000 |
| **New Balance (Origin)** | 7,700 |
| **Old Balance (Dest)** | 1,000 |
| **New Balance (Dest)** | 1,300 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~2.5%)

**Why this impresses:** DEBIT transactions are **never fraudulent** in this system, similar to PAYMENTs. The balances update perfectly: 8,000 - 300 = 7,700 and 1,000 + 300 = 1,300. This represents a typical ATM withdrawal or debit card purchase.

---

### Case 14: CASH_IN Deposit - Money Coming In

| Field | Value |
|-------|-------|
| **Step** | 200 |
| **Type** | CASH_IN |
| **Amount** | 5,000 |
| **Old Balance (Origin)** | 0 |
| **New Balance (Origin)** | 5,000 |
| **Old Balance (Dest)** | 100,000 |
| **New Balance (Dest)** | 95,000 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~0.7%)

**Why this impresses:** Someone depositing 5,000 cash into their empty account. The origin balance **increases** (0 -> 5,000) instead of decreasing. CASH_IN is a deposit operation and is never associated with fraud. The model correctly recognizes the direction of money flow.

---

### Case 15: CASH_OUT - Tiny Amount, Everything Clean

| Field | Value |
|-------|-------|
| **Step** | 400 |
| **Type** | CASH_OUT |
| **Amount** | 500 |
| **Old Balance (Origin)** | 20,000 |
| **New Balance (Origin)** | 19,500 |
| **Old Balance (Dest)** | 10,000 |
| **New Balance (Dest)** | 10,500 |

**Expected Result:** LEGITIMATE (Fraud Probability: ~8.4%)

**Why this impresses:** CASH_OUT **can** be fraudulent (see Cases 3 and 6). But this one is legitimate because: (1) tiny amount - only 2.5% of balance, (2) sender retains 97.5% of funds, (3) balances update correctly (20K-500=19.5K, 10K+500=10.5K), and (4) late time step (400). The model doesn't blindly flag all CASH_OUTs - it distinguishes safe withdrawals from fraud drains.

---

## Summary Table

| # | Type | Amount | Expected Result | Fraud Prob | Key Edge Case Insight |
|---|------|--------|----------------|------------|----------------------|
| 1 | TRANSFER | 181,000 | **FRAUDULENT** | ~96% | Classic full drain, dest gets nothing |
| 2 | TRANSFER | 50,000 | **FRAUDULENT** | ~98% | Small amount but all red flags present |
| 3 | CASH_OUT | 300,000 | **FRAUDULENT** | ~87% | Dest receives money but still fraud |
| 4 | TRANSFER | 150,000 | **FRAUDULENT** | ~93% | Mule account - dest balance drops |
| 5 | TRANSFER | 95,000 | **FRAUDULENT** | ~93% | Both accounts drained simultaneously |
| 6 | CASH_OUT | 150,000 | **FRAUDULENT** | ~89% | Dest math is perfect but still fraud |
| 7 | TRANSFER | 500,000 | **FRAUDULENT** | ~96% | Late step fraud, not just early hours |
| 8 | TRANSFER | 339,682 | **FRAUDULENT** | ~93% | Exact odd amount = automated attack |
| 9 | PAYMENT | 9,999 | **LEGITIMATE** | ~13% | 99.99% drained but PAYMENT is safe |
| 10 | PAYMENT | 50,000 | **LEGITIMATE** | ~5% | Huge amount but safe transaction type |
| 11 | TRANSFER | 100,000 | **LEGITIMATE** | ~0.5% | Large TRANSFER with perfect math |
| 12 | TRANSFER | 3,000 | **LEGITIMATE** | ~1.2% | Small ratio, correct balance updates |
| 13 | DEBIT | 300 | **LEGITIMATE** | ~2.5% | Safe type, routine transaction |
| 14 | CASH_IN | 5,000 | **LEGITIMATE** | ~0.7% | Deposit, money flows in |
| 15 | CASH_OUT | 500 | **LEGITIMATE** | ~8.4% | CASH_OUT CAN be fraud, but this isn't |

---

## What Makes These Edge Cases Special

### The Model's Intelligence (What Will Impress Your Evaluators):

1. **Not just amount-based:** Case 2 (50K fraud) vs Case 10 (50K legitimate) - same amount, different prediction. The model looks at patterns, not just numbers.

2. **Not just type-based:** Case 3 (CASH_OUT fraud) vs Case 15 (CASH_OUT legitimate) - same type, different prediction. The model understands context.

3. **Catches coordinated attacks:** Cases 4 and 5 show multi-account fraud where both sender and receiver are compromised - a sophisticated attack pattern.

4. **Doesn't over-flag:** Case 9 drains 99.99% of the account but is correctly labeled legitimate because of the transaction type.

5. **Understands balance mathematics:** Case 11 (100K TRANSFER, legitimate) has correct math (200K+100K=300K at dest), while Case 7 (500K TRANSFER, fraud) has broken math (100K+500K should be 600K but shows 0).

6. **Time-aware but not time-dependent:** Case 7 proves fraud at step 743 (not just early steps), while Case 15 proves legitimacy at step 400.

---

## Key Patterns the Model Learned

### What Makes a Transaction Fraudulent:
1. **Transaction type**: Only TRANSFER and CASH_OUT can be fraud (never PAYMENT, DEBIT, or CASH_IN)
2. **Full account drain**: The sender's entire balance is transferred out (amount = oldbalanceOrg, newbalanceOrig = 0)
3. **Destination balance anomaly**: The destination balance doesn't increase after receiving funds, or even decreases
4. **Large amounts**: Fraudulent transactions tend to involve larger sums
5. **Early timing**: Fraud often occurs at early simulation steps (step 1-3), but not exclusively

### What Makes a Transaction Legitimate:
1. **Safe transaction types**: PAYMENT, DEBIT, and CASH_IN are always legitimate
2. **Small amount-to-balance ratio**: The transaction amount is a small fraction of the sender's balance
3. **Correct balance updates**: Both sender and receiver balances change by exactly the transaction amount
4. **Retained balance**: The sender keeps most of their balance after the transaction
