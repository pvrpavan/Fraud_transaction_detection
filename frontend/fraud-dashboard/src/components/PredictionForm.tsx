import { useState } from 'react';
import { RefreshCw, Search } from 'lucide-react';

interface PredictionFormProps {
  onPredict: (data: Record<string, unknown>) => void;
  loading: boolean;
}

export function PredictionForm({ onPredict, loading }: PredictionFormProps) {
  const [formData, setFormData] = useState({
    step: '1',
    type: 'TRANSFER',
    amount: '181000',
    oldbalanceOrg: '181000',
    newbalanceOrig: '0',
    oldbalanceDest: '0',
    newbalanceDest: '0',
  });

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
      step: parseInt(formData.step),
      type: formData.type,
      amount: parseFloat(formData.amount),
      oldbalanceOrg: parseFloat(formData.oldbalanceOrg),
      newbalanceOrig: parseFloat(formData.newbalanceOrig),
      oldbalanceDest: parseFloat(formData.oldbalanceDest),
      newbalanceDest: parseFloat(formData.newbalanceDest),
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      {fields.map((field) => (
        <div key={field.key} className="group">
          <label className="text-xs text-gray-400 mb-1 block font-medium">{field.label}</label>
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
              value={formData[field.key as keyof typeof formData]}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              className="w-full bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 transition-all"
            />
          )}
        </div>
      ))}
      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 px-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg text-sm font-semibold text-white transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40 mt-4"
      >
        {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
        {loading ? 'Analyzing Transaction...' : 'Detect Fraud'}
      </button>
    </form>
  );
}
