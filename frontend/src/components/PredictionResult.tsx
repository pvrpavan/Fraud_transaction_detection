import { AlertTriangle, CheckCircle, Search } from 'lucide-react';

interface PredictionData {
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: string;
  confidence: number;
  risk_factors: string[];
  recommendation: string;
}

interface PredictionResultProps {
  prediction: PredictionData | null;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  if (!prediction) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-500">
        <div className="p-4 rounded-full bg-gray-800/40 mb-4">
          <Search className="w-10 h-10 opacity-30" />
        </div>
        <p className="text-sm font-medium">Enter transaction details and click &quot;Detect Fraud&quot;</p>
        <p className="text-xs mt-1 text-gray-600">Results will appear here in real-time</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fade-in">
      <div className={`rounded-xl p-6 text-center border-2 transition-all ${prediction.is_fraud ? 'bg-rose-500/10 border-rose-500/30 shadow-lg shadow-rose-500/10' : 'bg-emerald-500/10 border-emerald-500/30 shadow-lg shadow-emerald-500/10'}`}>
        <div className={`inline-flex p-3 rounded-full mb-3 ${prediction.is_fraud ? 'bg-rose-500/20' : 'bg-emerald-500/20'}`}>
          {prediction.is_fraud
            ? <AlertTriangle className="w-10 h-10 text-rose-400" />
            : <CheckCircle className="w-10 h-10 text-emerald-400" />
          }
        </div>
        <p className={`text-2xl font-bold ${prediction.is_fraud ? 'text-rose-400' : 'text-emerald-400'}`}>
          {prediction.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
        </p>
        <p className="text-sm text-gray-400 mt-1">
          Risk Level:{' '}
          <span className={`font-bold ${prediction.risk_level === 'HIGH' ? 'text-rose-400' : prediction.risk_level === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400'}`}>
            {prediction.risk_level}
          </span>
        </p>
      </div>

      <div className="bg-gray-800/40 rounded-lg p-4">
        <p className="text-xs text-gray-400 mb-3 font-medium">Fraud Probability</p>
        <div className="w-full bg-gray-700 rounded-full h-5 relative overflow-hidden">
          <div
            className={`h-5 rounded-full transition-all duration-700 ease-out ${prediction.fraud_probability > 0.7 ? 'bg-gradient-to-r from-rose-600 to-rose-400' : prediction.fraud_probability > 0.3 ? 'bg-gradient-to-r from-amber-600 to-amber-400' : 'bg-gradient-to-r from-emerald-600 to-emerald-400'}`}
            style={{ width: `${Math.max(prediction.fraud_probability * 100, 3)}%` }}
          />
          <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white drop-shadow-md">
            {(prediction.fraud_probability * 100).toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="bg-gray-800/40 rounded-lg p-4">
        <p className="text-xs text-gray-400 mb-1 font-medium">Model Confidence</p>
        <p className="text-lg font-bold text-white">{(prediction.confidence * 100).toFixed(2)}%</p>
      </div>

      {prediction.risk_factors && prediction.risk_factors.length > 0 && (
        <div className="bg-gray-800/40 rounded-lg p-4">
          <p className="text-xs text-gray-400 mb-3 font-medium">Risk Factors Identified</p>
          <ul className="space-y-2">
            {prediction.risk_factors.map((factor, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                <span className="text-gray-300">{factor}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {prediction.recommendation && (
        <div className={`rounded-lg p-4 border ${prediction.is_fraud ? 'bg-rose-500/5 border-rose-500/20' : 'bg-emerald-500/5 border-emerald-500/20'}`}>
          <p className="text-xs text-gray-400 mb-1 font-medium">Recommendation</p>
          <p className={`text-sm font-semibold ${prediction.is_fraud ? 'text-rose-300' : 'text-emerald-300'}`}>
            {prediction.recommendation}
          </p>
        </div>
      )}
    </div>
  );
}
