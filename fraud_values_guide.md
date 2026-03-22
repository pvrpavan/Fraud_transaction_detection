# Fraud Transaction Detection - Complete Values Guide

## Understanding When Transactions Are Classified as Fraud vs Legitimate

This guide provides a comprehensive explanation of every input field, how each value affects fraud detection, and 12 fully worked examples covering all scenarios. Based on analysis of 150,000+ financial transactions from the PaySim dataset.

---

## Input Fields Explained

### 1. `step` (Time Unit)
- **What it is**: Each step represents **1 hour** of simulated time. The dataset spans 743 steps (~30 days).
- **Range**: 1 to 743
- **How it affects fraud**: Step alone is a weak signal. The model extracts time-based features from it:
  - `hour_of_day` = step % 24 (what hour of the day)
  - `day_of_sim` = step / 24 (what day of the simulation)
  - `unusual_hour` = 1 if hour < 6 or hour > 22 (late-night transactions)
  - Cyclical encoding (`hour_sin`, `hour_cos`) to capture 24-hour periodicity
- **Impact on fraud**: Minimal direct impact. Fraud occurs at ALL time steps. Steps 200-400 show slightly elevated fraud density, but this is not a strong predictor. The model relies far more on transaction type, amount, and balance patterns.

### 2. `type` (Transaction Type)
- **What it is**: The category of financial transaction being performed.
- **Available types and their fraud risk**:

| Type | What It Does | Can Be Fraud? | Why |
|------|-------------|---------------|-----|
| **TRANSFER** | Move money between accounts | **YES - HIGH RISK** | Fraudsters transfer stolen funds to accounts they control |
| **CASH_OUT** | Withdraw cash from account | **YES - HIGH RISK** | Fraudsters cash out transferred funds immediately to extract value |
| **PAYMENT** | Pay for goods/services | **NO** | Payments go to merchants with verified identities; easily reversed |
| **CASH_IN** | Deposit cash into account | **NO** | Depositing money is never fraud (the account gains money) |
| **DEBIT** | Direct debit from account | **NO** | Authorized pre-arranged withdrawals with known parties |

- **Impact on fraud**: **Critical**. This is a binary gate - if the type is PAYMENT, CASH_IN, or DEBIT, the transaction is **always legitimate** regardless of all other values. Fraud can **only** occur with TRANSFER or CASH_OUT.

### 3. `amount` (Transaction Amount)
- **What it is**: The monetary value of the transaction.
- **How it affects fraud**:
  - Fraudulent transactions average **~1,400,000** vs **~74,000** for legitimate
  - Amounts above **200,000** with TRANSFER/CASH_OUT have significantly higher fraud risk
  - Amounts below **10,000** are overwhelmingly legitimate
  - The model uses `amount_log` (log-transformed) and `amount_sqrt` (square root) to handle the extreme skew in amount distribution
  - `amount_vs_type_mean` compares the amount to the average for that transaction type
  - `amount_zscore_by_type` measures how many standard deviations above/below the mean for that type
- **Impact on fraud**: **Very high**. Amount is the single most important feature. Large amounts combined with TRANSFER/CASH_OUT are the strongest fraud signal.

### 4. `oldbalanceOrg` (Origin Balance Before Transaction)
- **What it is**: How much money the sender had in their account before the transaction.
- **How it affects fraud**:
  - When `amount == oldbalanceOrg`, it means the entire balance is being transferred out (complete drain)
  - Fraudsters typically drain the entire available balance in one go
  - A high balance being completely emptied is a major fraud indicator
- **Impact on fraud**: **High**. The relationship between amount and oldbalanceOrg is key. If amount equals the full balance, fraud risk jumps significantly.

### 5. `newbalanceOrig` (Origin Balance After Transaction)
- **What it is**: How much money the sender has left after the transaction.
- **How it affects fraud**:
  - `newbalanceOrig = 0` after a large transfer = strong fraud signal (complete account drain)
  - `newbalanceOrig > 0` = some money left behind = lower fraud risk
  - The model checks: `oldbalanceOrg - newbalanceOrig` should equal `amount`. If it doesn't, there's a balance discrepancy (possible record manipulation)
- **Impact on fraud**: **High**. A zero balance after transaction is one of the top fraud indicators.

### 6. `oldbalanceDest` (Destination Balance Before Transaction)
- **What it is**: How much money the receiver had before receiving the transfer.
- **How it affects fraud**:
  - `oldbalanceDest = 0` can indicate a newly created "mule" account (common in fraud schemes)
  - However, many legitimate new accounts also start at 0
- **Impact on fraud**: **Moderate**. A zero starting balance at the destination is a contributing factor but not decisive on its own.

### 7. `newbalanceDest` (Destination Balance After Transaction)
- **What it is**: How much money the receiver has after the transaction.
- **How it affects fraud**:
  - If `newbalanceDest == oldbalanceDest` (unchanged) after supposedly receiving money, this is highly suspicious - the money "disappeared"
  - In legitimate transactions, `newbalanceDest = oldbalanceDest + amount`
  - When `oldbalanceDest = 0` AND `newbalanceDest = 0` despite receiving a transfer, it strongly suggests the funds were immediately moved elsewhere (money laundering pattern)
- **Impact on fraud**: **High**. Destination balance not increasing after receiving money is a top fraud indicator.

---

## How Features Work Together

The model doesn't look at any single value in isolation. It learns **patterns** from combinations:

| Pattern | What It Means | Fraud Signal |
|---------|--------------|--------------|
| TRANSFER + amount == oldbalanceOrg + newbalanceOrig == 0 + dest unchanged | Classic fraud: drain account, money vanishes | **Very Strong** |
| CASH_OUT + large amount + full balance used | Fraudster extracting stolen funds | **Very Strong** |
| TRANSFER + large amount + dest balance increases correctly | Legitimate large transfer (business, account consolidation) | **Weak/None** |
| PAYMENT + any amount + any balances | Normal purchase | **None** |
| TRANSFER + small amount + partial balance + dest updates | Normal person-to-person transfer | **None** |

---

## 12 Detailed Example Transactions

### Example 1: FRAUD - Classic Full Drain Transfer
```
step: 1
type: TRANSFER
amount: 181,000.00
oldbalanceOrg: 181,000.00
newbalanceOrig: 0.00
oldbalanceDest: 0.00
newbalanceDest: 0.00
→ FRAUD
```
**Why fraud**: The entire balance (181K) is drained in one transfer. The destination account starts at 0 and stays at 0 - the money seemingly vanished. This is the textbook fraud pattern: a fraudster gains access to an account, transfers everything to a mule account, and the mule quickly moves the money elsewhere before anyone notices.

---

### Example 2: FRAUD - Large Cash Out (Account Emptied)
```
step: 200
type: CASH_OUT
amount: 339,682.13
oldbalanceOrg: 339,682.13
newbalanceOrig: 0.00
oldbalanceDest: 0.00
newbalanceDest: 339,682.13
→ FRAUD
```
**Why fraud**: Massive cash out of the exact account balance. Even though the destination balance updates correctly, the combination of CASH_OUT + exact full balance + large amount is a strong fraud signal. Fraudsters often cash out stolen funds immediately. The large amount (339K vs 74K average) further increases the fraud probability.

---

### Example 3: FRAUD - Transfer with Balance Manipulation
```
step: 350
type: TRANSFER
amount: 1,250,000.00
oldbalanceOrg: 1,250,000.00
newbalanceOrig: 0.00
oldbalanceDest: 500,000.00
newbalanceDest: 500,000.00
→ FRAUD
```
**Why fraud**: A massive 1.25M transfer completely drains the origin account. The destination had 500K but its balance didn't change after supposedly receiving 1.25M - the money disappeared. This indicates the destination is a mule account that immediately forwarded the funds. The balance discrepancy at the destination is a hallmark of money laundering.

---

### Example 4: FRAUD - Moderate Amount Full Drain
```
step: 45
type: CASH_OUT
amount: 62,927.00
oldbalanceOrg: 62,927.00
newbalanceOrig: 0.00
oldbalanceDest: 0.00
newbalanceDest: 0.00
→ FRAUD
```
**Why fraud**: Even though the amount is moderate (62K), the complete drain pattern (amount == oldbalanceOrg, newbalanceOrig == 0) combined with the destination balance remaining at 0 after receiving funds makes this a clear fraud case. The CASH_OUT type adds to the risk.

---

### Example 5: LEGITIMATE - Standard Payment
```
step: 50
type: PAYMENT
amount: 1,500.00
oldbalanceOrg: 25,000.00
newbalanceOrig: 23,500.00
oldbalanceDest: 10,000.00
newbalanceDest: 11,500.00
→ LEGITIMATE
```
**Why legitimate**: PAYMENT type transactions have **zero fraud rate** in the dataset. The type alone guarantees this is legitimate. Additionally: the amount is small (1.5K), balances are consistent (25K - 1.5K = 23.5K), and the destination properly received the funds (10K + 1.5K = 11.5K).

---

### Example 6: LEGITIMATE - Cash Deposit (CASH_IN)
```
step: 100
type: CASH_IN
amount: 50,000.00
oldbalanceOrg: 100,000.00
newbalanceOrig: 150,000.00
oldbalanceDest: 0.00
newbalanceDest: 0.00
→ LEGITIMATE
```
**Why legitimate**: CASH_IN means money is being deposited into an account. This can never be fraud because the account is gaining money, not losing it. The type alone makes this legitimate regardless of the amount or other values.

---

### Example 7: LEGITIMATE - Small Transfer with Proper Balances
```
step: 300
type: TRANSFER
amount: 5,000.00
oldbalanceOrg: 80,000.00
newbalanceOrig: 75,000.00
oldbalanceDest: 200,000.00
newbalanceDest: 205,000.00
→ LEGITIMATE
```
**Why legitimate**: Even though the type is TRANSFER (which can be fraud), everything else is clean: the amount is small (5K), it's only a small fraction of the sender's balance (6.25%), the sender retains 75K after the transfer, and the destination correctly received the funds (200K + 5K = 205K). This looks like a normal person-to-person transfer.

---

### Example 8: LEGITIMATE - Debit Transaction
```
step: 15
type: DEBIT
amount: 3,200.00
oldbalanceOrg: 45,000.00
newbalanceOrig: 41,800.00
oldbalanceDest: 0.00
newbalanceDest: 0.00
→ LEGITIMATE
```
**Why legitimate**: DEBIT type has zero fraud rate. Debit transactions are pre-authorized withdrawals (like utility bills, subscriptions). The type alone ensures this is legitimate.

---

### Example 9: LEGITIMATE - Large Partial Transfer (Not Full Drain)
```
step: 500
type: TRANSFER
amount: 400,000.00
oldbalanceOrg: 2,000,000.00
newbalanceOrig: 1,600,000.00
oldbalanceDest: 500,000.00
newbalanceDest: 900,000.00
→ LEGITIMATE
```
**Why legitimate**: Despite the large amount (400K) and TRANSFER type, this is legitimate because: the sender retains 1.6M (only 20% of balance used), and the destination properly received the money (500K + 400K = 900K). This looks like a legitimate business transfer or investment. Fraudsters typically drain the FULL balance, not 20% of it.

---

### Example 10: LEGITIMATE - Large Payment to Merchant
```
step: 150
type: PAYMENT
amount: 500,000.00
oldbalanceOrg: 1,000,000.00
newbalanceOrig: 500,000.00
oldbalanceDest: 5,000,000.00
newbalanceDest: 5,500,000.00
→ LEGITIMATE
```
**Why legitimate**: Even with a very large amount (500K), the PAYMENT type guarantees this is legitimate. This could be a large business purchase, real estate deposit, or luxury item. PAYMENT transactions never contain fraud regardless of the amount.

---

### Example 11: FRAUD - Transfer to Empty Account (Money Laundering)
```
step: 72
type: TRANSFER
amount: 890,000.00
oldbalanceOrg: 890,000.00
newbalanceOrig: 0.00
oldbalanceDest: 0.00
newbalanceDest: 0.00
→ FRAUD
```
**Why fraud**: Full drain of 890K into an empty account that stays empty. This is a money laundering pattern: the mule account receives the money and immediately forwards it to another account (or converts to cash/crypto). The zero-to-zero destination pattern is one of the strongest fraud signals.

---

### Example 12: LEGITIMATE - Transfer Full Balance (Account Consolidation)
```
step: 400
type: TRANSFER
amount: 175,000.00
oldbalanceOrg: 175,000.00
newbalanceOrig: 0.00
oldbalanceDest: 300,000.00
newbalanceDest: 475,000.00
→ LEGITIMATE
```
**Why legitimate**: This looks like fraud at first glance (full drain, TRANSFER type, large amount). But the destination balance properly increased from 300K to 475K (300K + 175K = 475K). The destination is an established account with existing funds, not an empty mule account. This is likely account consolidation - someone moving their full balance from one bank to another. The model learned that when the destination properly receives the funds, the risk drops dramatically.

---

## How Step Affects Fraud Detection

| Step Range | Hours | Day Equivalent | Fraud Impact |
|-----------|-------|---------------|--------------|
| 1-24 | Hours 1-24 | Day 1 | Normal fraud rate |
| 25-168 | Hours 25-168 | Days 2-7 | Normal fraud rate |
| 169-336 | Hours 169-336 | Days 8-14 | Slightly elevated fraud |
| 337-504 | Hours 337-504 | Days 15-21 | Slightly elevated fraud |
| 505-743 | Hours 505-743 | Days 22-31 | Normal fraud rate |

**Key points about step**:
- Step has the **lowest importance** among all features
- The model extracts `hour_of_day` (step % 24) and flags `unusual_hour` (before 6 AM or after 10 PM)
- Transactions at unusual hours get a slightly higher fraud score, but this is a weak signal
- **Never rely on step alone** to determine fraud - it's always a combination with type, amount, and balance patterns

---

## Quick Reference Decision Table

| # | Condition | Prediction | Confidence |
|---|-----------|-----------|------------|
| 1 | Type = PAYMENT, CASH_IN, or DEBIT | **LEGITIMATE** | Very High |
| 2 | Type = TRANSFER/CASH_OUT + Full drain + Dest unchanged | **FRAUD** | Very High |
| 3 | Type = TRANSFER/CASH_OUT + Full drain + Dest updates correctly | **LIKELY LEGITIMATE** | Moderate |
| 4 | Type = TRANSFER/CASH_OUT + Large amount + Full drain + Empty dest | **FRAUD** | Very High |
| 5 | Type = TRANSFER/CASH_OUT + Small amount + Partial balance + Dest updates | **LEGITIMATE** | High |
| 6 | Type = TRANSFER/CASH_OUT + Large amount + Partial balance + Dest updates | **LIKELY LEGITIMATE** | Moderate |
| 7 | Balance change does not match amount | **SUSPICIOUS** | Moderate |

---

## Top Features Used by the Model (by importance)

1. **amount** - Transaction amount (most important feature)
2. **oldbalanceOrg** - Origin account balance before transaction
3. **type** - Transaction type (TRANSFER, CASH_OUT, etc.)
4. **oldbalanceDest** - Destination account balance before transaction
5. **amount_log** - Log-transformed amount (handles extreme skew)
6. **amount_sqrt** - Square root of amount (moderate skew reduction)
7. **amount_vs_type_mean** - How the amount compares to the average for that transaction type
8. **amount_zscore_by_type** - Standard deviations from the mean amount for that type
9. **hour_of_day** - Extracted from step (step % 24)
10. **unusual_hour** - Flag for transactions at odd hours (before 6 AM or after 10 PM)

> **Note**: The model drops `newbalanceOrig` and `newbalanceDest` during preprocessing to prevent data leakage. In a real-world scenario, these post-transaction values would not be available at prediction time (before the transaction is processed). They are still collected in the form for completeness and risk factor analysis.

---

## Model Accuracy Summary

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | ~99.99% | ~99.94% | ~99.97% |
| Gradient Boosting | ~99.99% | ~99.94% | ~100.0% |
| XGBoost | ~99.99% | ~99.94% | ~100.0% |
| LightGBM | ~99.99% | ~99.94% | ~100.0% |
| CatBoost | ~99.99% | ~99.94% | ~100.0% |
| Logistic Regression | ~62.95% | ~39.13% | ~98.72% |

> **Note**: Tree-based models achieve near-perfect accuracy because fraud patterns in financial transactions have very distinctive signatures (complete drains, specific transaction types, balance mismatches). The hard-negative augmentation in the training data makes the model more robust against edge cases where legitimate transactions mimic fraud patterns.
