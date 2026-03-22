import { Search, Eye } from 'lucide-react';
import { SectionHeader } from './SectionHeader';
import { PredictionForm } from './PredictionForm';
import { PredictionResult } from './PredictionResult';

interface PredictionData {
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: string;
  confidence: number;
  risk_factors: string[];
  recommendation: string;
}

interface PredictTabProps {
  prediction: PredictionData | null;
  predLoading: boolean;
  onPredict: (data: Record<string, unknown>) => void;
}

export function PredictTab({ prediction, predLoading, onPredict }: PredictTabProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
      <div className="glass-card p-5">
        <SectionHeader icon={Search} title="Transaction Analyzer" subtitle="Enter transaction details to check for fraud" />
        <PredictionForm onPredict={onPredict} loading={predLoading} />
      </div>
      <div className="glass-card p-5">
        <SectionHeader icon={Eye} title="Analysis Result" subtitle="Real-time fraud probability assessment" />
        <PredictionResult prediction={prediction} />
      </div>
    </div>
  );
}
