# Fraud Transaction Detection - Values Guide

## Understanding When Transactions Are Classified as Fraud vs Legitimate

This guide explains the key indicators our ML models use to identify fraudulent transactions based on analysis of 150,000 real financial transactions.

---

## Transaction Types and Fraud Risk

| Transaction Type | Fraud Risk | Details |
|-----------------|------------|---------|
| **TRANSFER** | **HIGH** | Most common fraud type. Fraudsters transfer money to accounts they control. |
| **CASH_OUT** | **HIGH** | Second most common fraud type. Fraudsters cash out transferred funds immediately. |
| **PAYMENT** | **NONE** | No fraud detected in payment transactions in our dataset. |
| **CASH_IN** | **NONE** | No fraud detected in cash-in transactions. |
| **DEBIT** | **NONE** | No fraud detected in debit transactions. |

> **Key Insight**: Fraud ONLY occurs in TRANSFER and CASH_OUT transaction types.

---

## Fraud Indicators (Values That Suggest FRAUD)

### 1. Complete Account Drain
- **Old Balance (Origin) > 0** AND **New Balance (Origin) = 0**
- The entire account balance is transferred out in a single transaction
- Example: `oldbalanceOrg = 181,000` → `newbalanceOrig = 0`

### 2. Large Transaction Amounts
- Fraudulent transactions tend to have **higher amounts** than legitimate ones
- Average fraud amount: **~1,400,000** (vs ~74,000 for legitimate)
- Transactions above **200,000** have significantly higher fraud risk

### 3. Balance Mismatch at Destination
- **Old Balance (Dest) = 0** AND **New Balance (Dest) = 0** after receiving money
- This indicates the destination account shows no record of receiving the funds
- Common in money laundering schemes where funds are quickly moved

### 4. Amount Equals Origin Balance
- When `amount ≈ oldbalanceOrg` (transaction drains the full balance)
- Fraudsters typically take everything available

### 5. Balance Change Discrepancy
- When `oldbalanceOrg - newbalanceOrig ≠ amount`
- The balance change doesn't match the transaction amount
- Indicates potential manipulation of account records

### 6. Transaction Type is TRANSFER or CASH_OUT
- These are the ONLY two types where fraud occurs
- TRANSFER followed by CASH_OUT is a classic fraud pattern

### 7. High Step Values (Time)
- Fraud can occur at any time step, but certain periods show higher fraud density
- Steps in the range of 200-400 show slightly elevated fraud rates

---

## Legitimate Transaction Indicators (Values That Suggest NOT FRAUD)

### 1. Transaction Type is PAYMENT, CASH_IN, or DEBIT
- **Zero fraud rate** for these transaction types
- These are safe transaction categories

### 2. Small Transaction Amounts
- Amounts below **10,000** are overwhelmingly legitimate
- Micro-transactions (< 1,000) have near-zero fraud risk

### 3. Consistent Balance Changes
- `oldbalanceOrg - newbalanceOrig = amount` (exact match)
- The balance change correctly reflects the transaction amount

### 4. Destination Receives Funds Properly
- `newbalanceDest - oldbalanceDest = amount` (for transfers)
- Money properly appears in the destination account

### 5. Partial Balance Transactions
- When `newbalanceOrig > 0` (money remains in origin account)
- Fraudsters typically drain accounts completely

### 6. Normal Amount Ranges
- Transaction amounts within typical ranges for the account
- Amounts proportional to account balance (not draining it)

---

## Quick Reference Decision Table

| Condition | Prediction |
|-----------|-----------|
| Type = PAYMENT, CASH_IN, or DEBIT | **LEGITIMATE** |
| Type = TRANSFER/CASH_OUT + Amount > 200,000 + Full drain | **VERY LIKELY FRAUD** |
| Type = TRANSFER + Dest balance unchanged | **LIKELY FRAUD** |
| Type = CASH_OUT + Amount = Full balance | **LIKELY FRAUD** |
| Type = TRANSFER/CASH_OUT + Small amount + Partial balance | **LIKELY LEGITIMATE** |
| Balance change matches amount exactly | **LIKELY LEGITIMATE** |

---

## Top Features Used by the Model (by importance)

1. **amount** - Transaction amount (most important feature)
2. **oldbalanceOrg** - Origin account balance before transaction
3. **newbalanceOrig** - Origin account balance after transaction
4. **oldbalanceDest** - Destination account balance before transaction
5. **newbalanceDest** - Destination account balance after transaction
6. **type** - Transaction type (TRANSFER, CASH_OUT, etc.)
7. **balance_change_orig** - Calculated: oldbalanceOrg - newbalanceOrig
8. **balance_change_dest** - Calculated: newbalanceDest - oldbalanceDest
9. **amount_balance_ratio** - Ratio of amount to origin balance
10. **is_complete_drain** - Whether the transaction drains the full balance

---

## Example Transactions

### Fraudulent Transaction Example
```
step: 1
type: TRANSFER
amount: 181,000.00
oldbalanceOrg: 181,000.00
newbalanceOrig: 0.00
oldbalanceDest: 0.00
newbalanceDest: 0.00
→ FRAUD (complete drain, destination unchanged, high amount)
```

### Legitimate Transaction Example
```
step: 50
type: PAYMENT
amount: 1,500.00
oldbalanceOrg: 25,000.00
newbalanceOrig: 23,500.00
oldbalanceDest: 10,000.00
newbalanceDest: 11,500.00
→ LEGITIMATE (payment type, small amount, balances consistent)
```

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

> **Note**: Tree-based models achieve near-perfect accuracy because fraud patterns in financial transactions have very distinctive signatures (complete drains, specific transaction types, balance mismatches).
