import { Search, Eye, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
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

const QUICK_EXAMPLES = [
  {
    label: 'Fraud: Full Drain Transfer',
    icon: AlertTriangle,
    color: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    data: { step: 1, type: 'TRANSFER', amount: 181000, oldbalanceOrg: 181000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 0 },
  },
  {
    label: 'Fraud: Large Cash Out',
    icon: AlertTriangle,
    color: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    data: { step: 200, type: 'CASH_OUT', amount: 339682, oldbalanceOrg: 339682, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 0 },
  },
  {
    label: 'Legit: Small Payment',
    icon: CheckCircle,
    color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    data: { step: 50, type: 'PAYMENT', amount: 1500, oldbalanceOrg: 25000, newbalanceOrig: 23500, oldbalanceDest: 10000, newbalanceDest: 11500 },
  },
  {
    label: 'Legit: Cash In',
    icon: CheckCircle,
    color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    data: { step: 100, type: 'CASH_IN', amount: 50000, oldbalanceOrg: 100000, newbalanceOrig: 150000, oldbalanceDest: 0, newbalanceDest: 0 },
  },
];

export function PredictTab({ prediction, predLoading, onPredict }: PredictTabProps) {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Quick Examples */}
      <div className="glass-card p-5">
        <SectionHeader icon={Zap} title="Quick Examples" subtitle="Click to auto-fill and test common scenarios" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {QUICK_EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              onClick={() => onPredict(ex.data)}
              disabled={predLoading}
              className={`${ex.color} border rounded-lg p-3 text-left transition-all duration-200 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <div className="flex items-center gap-2 mb-1">
                <ex.icon className="w-3.5 h-3.5" />
                <span className="text-xs font-semibold">{ex.label}</span>
              </div>
              <p className="text-[10px] text-gray-400">
                {ex.data.type} - ${ex.data.amount.toLocaleString()}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Main Form + Result */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-5">
          <SectionHeader icon={Search} title="Transaction Analyzer" subtitle="Enter transaction details to check for fraud" />
          <PredictionForm onPredict={onPredict} loading={predLoading} />
        </div>
        <div className="glass-card p-5">
          <SectionHeader icon={Eye} title="Analysis Result" subtitle="Real-time fraud probability assessment" />
          <PredictionResult prediction={prediction} loading={predLoading} />
        </div>
      </div>
    </div>
  );
}
