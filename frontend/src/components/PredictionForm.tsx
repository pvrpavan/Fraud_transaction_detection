import { useState, useCallback } from 'react';
import { RefreshCw, Search, RotateCcw, Info } from 'lucide-react';

interface PredictionFormProps {
  onPredict: (data: Record<string, unknown>) => void;
  loading: boolean;
}

const DEFAULT_FORM = {
  step: '1',
  type: 'TRANSFER',
  amount: '',
  oldbalanceOrg: '',
  newbalanceOrig: '',
  oldbalanceDest: '',
  newbalanceDest: '',
};

const FIELD_HINTS: Record<string, string> = {
  step: 'Time unit (1 step = 1 hour). Range: 1-743. Fraud occurs at all times.',
  type: 'Only TRANSFER and CASH_OUT can be fraud. PAYMENT/DEBIT/CASH_IN are always legitimate.',
  amount: 'Transaction amount. Fraud averages ~1.4M vs ~74K for legitimate.',
  oldbalanceOrg: 'Sender balance before transaction.',
  newbalanceOrig: 'Sender balance after transaction. 0 = full drain (fraud signal).',
  oldbalanceDest: 'Receiver balance before transaction.',
  newbalanceDest: 'Receiver balance after transaction. Unchanged = suspicious.',
};

export function PredictionForm({ onPredict, loading }: PredictionFormProps) {
  const [formData, setFormData] = useState(DEFAULT_FORM);
  const [showHint, setShowHint] = useState<string | null>(null);

  const handleReset = useCallback(() => {
    setFormData(DEFAULT_FORM);
    setShowHint(null);
  }, []);

  const fields = [
    { key: 'step', label: 'Step (Time)', type: 'number', icon: '⏱' },
    { key: 'type', label: 'Transaction Type', type: 'select', options: ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], icon: '📋' },
    { key: 'amount', label: 'Amount', type: 'number', icon: '💰' },
    { key: 'oldbalanceOrg', label: 'Old Balance (Origin)', type: 'number', icon: '🏦' },
    { key: 'newbalanceOrig', label: 'New Balance (Origin)', type: 'number', icon: '🏦' },
    { key: 'oldbalanceDest', label: 'Old Balance (Dest)', type: 'number', icon: '🎯' },
    { key: 'newbalanceDest', label: 'New Balance (Dest)', type: 'number', icon: '🎯' },
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict({
      step: parseInt(formData.step) || 1,
      type: formData.type,
      amount: parseFloat(formData.amount) || 0,
      oldbalanceOrg: parseFloat(formData.oldbalanceOrg) || 0,
      newbalanceOrig: parseFloat(formData.newbalanceOrig) || 0,
      oldbalanceDest: parseFloat(formData.oldbalanceDest) || 0,
      newbalanceDest: parseFloat(formData.newbalanceDest) || 0,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      {fields.map((field) => (
        <div key={field.key} className="group">
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs text-gray-400 font-medium">{field.label}</label>
            <button
              type="button"
              className="text-gray-600 hover:text-indigo-400 transition-colors p-0.5"
              onClick={() => setShowHint(showHint === field.key ? null : field.key)}
              title="Show hint"
            >
              <Info className="w-3 h-3" />
            </button>
          </div>
          {showHint === field.key && (
            <p className="text-[11px] text-indigo-300/70 mb-1.5 bg-indigo-500/5 border border-indigo-500/10 rounded px-2 py-1">
              {FIELD_HINTS[field.key]}
            </p>
          )}
          {field.type === 'select' ? (
            <select
              value={formData[field.key as keyof typeof formData]}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              className="w-full bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 transition-all"
            >
              {field.options?.map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          ) : (
            <input
              type="number"
              step="any"
              placeholder="0"
              value={formData[field.key as keyof typeof formData]}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              className="w-full bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 transition-all"
            />
          )}
        </div>
      ))}
      <div className="flex gap-3 mt-4">
        <button
          type="submit"
          disabled={loading}
          className="flex-1 py-3 px-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg text-sm font-semibold text-white transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40"
        >
          {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          {loading ? 'Analyzing...' : 'Detect Fraud'}
        </button>
        <button
          type="button"
          onClick={handleReset}
          disabled={loading}
          className="py-3 px-4 bg-gray-800/60 border border-gray-700 hover:border-gray-500 rounded-lg text-sm font-medium text-gray-300 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          title="Reset form"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>
    </form>
  );
}
