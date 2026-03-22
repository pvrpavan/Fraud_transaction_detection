import { BarChart3, Brain, Zap, ChevronDown, ChevronUp } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { SectionHeader } from './SectionHeader';
import type { SummaryData } from '../types';

interface ModelsTabProps {
  summary: SummaryData;
  expandedModel: string | null;
  onToggleModel: (model: string) => void;
}

export function ModelsTab({ summary, expandedModel, onToggleModel }: ModelsTabProps) {
  const modelComparisonData = summary.model_comparison.map((m) => ({
    name: m.model.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    Precision: parseFloat((m.precision * 100).toFixed(2)),
    Recall: parseFloat((m.recall * 100).toFixed(2)),
    'F1 Score': parseFloat((m.f1 * 100).toFixed(2)),
    'ROC AUC': parseFloat((m.roc_auc * 100).toFixed(2)),
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="glass-card p-5">
        <SectionHeader icon={BarChart3} title="Model Performance Comparison" subtitle="Side-by-side metric comparison" />
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={modelComparisonData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-30} textAnchor="end" height={80} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Bar dataKey="Precision" fill="#6366F1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Recall" fill="#EC4899" radius={[4, 4, 0, 0]} />
            <Bar dataKey="F1 Score" fill="#10B981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="ROC AUC" fill="#F59E0B" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {summary.model_comparison.map((model) => (
          <div
            key={model.model}
            className={`glass-card p-4 cursor-pointer transition-all duration-300 hover:scale-[1.02] ${model.is_best ? 'ring-2 ring-indigo-500/50 glow-indigo' : 'hover:border-gray-700'}`}
            onClick={() => onToggleModel(model.model)}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-bold text-white capitalize">{model.model.replace(/_/g, ' ')}</h3>
                {model.is_best && (
                  <span className="text-xs bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded-full border border-indigo-500/30 inline-block mt-1">
                    Best Model
                  </span>
                )}
              </div>
              {expandedModel === model.model
                ? <ChevronUp className="w-4 h-4 text-gray-400" />
                : <ChevronDown className="w-4 h-4 text-gray-400" />
              }
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gray-800/40 rounded-lg p-2 text-center">
                <p className="text-xs text-gray-500">F1</p>
                <p className="text-sm font-bold text-emerald-400">{(model.f1 * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-gray-800/40 rounded-lg p-2 text-center">
                <p className="text-xs text-gray-500">AUC</p>
                <p className="text-sm font-bold text-purple-400">{(model.roc_auc * 100).toFixed(2)}%</p>
              </div>
            </div>
            {expandedModel === model.model && (
              <div className="mt-3 pt-3 border-t border-gray-800 space-y-2 text-xs animate-fade-in">
                <div className="flex justify-between"><span className="text-gray-400">Precision</span><span className="text-white">{(model.precision * 100).toFixed(2)}%</span></div>
                <div className="flex justify-between"><span className="text-gray-400">Recall</span><span className="text-white">{(model.recall * 100).toFixed(2)}%</span></div>
                <div className="flex justify-between"><span className="text-gray-400">CV Mean</span><span className="text-white">{(model.cv_mean * 100).toFixed(2)}%</span></div>
                <div className="flex justify-between"><span className="text-gray-400">Training Time</span><span className="text-white">{model.training_time.toFixed(1)}s</span></div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="glass-card p-5">
        <SectionHeader icon={Zap} title="Hyperparameter Tuning" subtitle="Optimized parameters for best model" />
        <div className="bg-gray-800/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-4 h-4 text-indigo-400" />
            <span className="text-sm font-semibold text-white capitalize">
              {summary.hyperparameter_tuning.model?.replace(/_/g, ' ')}
            </span>
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
              CV Score: {(summary.hyperparameter_tuning.best_cv_score * 100).toFixed(2)}%
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {Object.entries(summary.hyperparameter_tuning.best_params || {}).map(([key, value]) => (
              <div key={key} className="bg-gray-800/60 rounded-lg p-2 hover:bg-gray-800/80 transition-colors">
                <p className="text-xs text-gray-500">{key}</p>
                <p className="text-sm text-white font-mono">{String(value)}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
