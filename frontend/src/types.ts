export interface SummaryData {
  dataset: {
    total_transactions: number;
    fraud_transactions: number;
    legitimate_transactions: number;
    fraud_ratio: number;
  };
  sampling: {
    strategy: string;
    original_rows: number;
    sampled_rows: number;
    fraud_count: number;
    fraud_ratio: number;
  };
  preprocessing: {
    n_features: number;
    feature_names: string[];
  };
  imbalance_handling: {
    method: string;
    original_train_size: number;
    balanced_train_size: number;
    original_fraud_ratio: number;
    balanced_fraud_ratio: number;
  };
  model_comparison: ModelResult[];
  best_model: {
    name: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number;
    specificity: number;
    mcc: number;
  };
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
  hyperparameter_tuning: {
    model: string;
    best_params: Record<string, unknown>;
    best_cv_score: number;
  };
  pipeline_time: number;
}

export interface ModelResult {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  cv_mean: number;
  cv_std: number;
  training_time: number;
  is_best: boolean;
}

export interface FeatureData {
  feature_importance: Record<string, number>;
  top_features: string[];
}

export interface PlotInfo {
  filename: string;
  url: string;
  size: number;
  title: string;
}

export interface PredictionResult {
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: string;
  confidence: number;
  risk_factors: string[];
  recommendation: string;
}
