import { BarChart3, Brain, Zap, ChevronDown, ChevronUp, Award, Clock, Trophy } from 'lucide-react';
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

  const sortedModels = [...summary.model_comparison].sort((a, b) => b.f1 - a.f1);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Chart */}
      <div className="glass-card p-6">
        <SectionHeader icon={BarChart3} title="Model Performance Comparison" subtitle="Side-by-side metric comparison across all trained models" />
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={modelComparisonData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-30} textAnchor="end" height={80} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', color: '#e2e8f0', boxShadow: '0 10px 40px rgba(0,0,0,0.3)' }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Bar dataKey="Precision" fill="#6366F1" radius={[4, 4, 0, 0]} animationDuration={800} />
            <Bar dataKey="Recall" fill="#EC4899" radius={[4, 4, 0, 0]} animationDuration={1000} />
            <Bar dataKey="F1 Score" fill="#10B981" radius={[4, 4, 0, 0]} animationDuration={1200} />
            <Bar dataKey="ROC AUC" fill="#F59E0B" radius={[4, 4, 0, 0]} animationDuration={1400} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Model Cards */}
      <div>
        <div className="flex items-center gap-2 mb-4">
          <Trophy className="w-5 h-5 text-amber-400" />
          <h3 className="text-lg font-bold text-white">Model Rankings</h3>
          <span className="text-xs text-gray-500 ml-2">Sorted by F1 Score</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sortedModels.map((model, index) => (
            <div
              key={model.model}
              className={`glass-card p-5 cursor-pointer transition-all duration-300 hover:scale-[1.02] relative overflow-hidden ${model.is_best ? 'ring-2 ring-indigo-500/50 glow-indigo' : 'hover:border-gray-700'}`}
              onClick={() => onToggleModel(model.model)}
              style={{ animationDelay: `${index * 80}ms` }}
            >
              {model.is_best && (
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-indigo-500/10 to-transparent rounded-bl-full" />
              )}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2.5">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold ${
                    index === 0 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                    index === 1 ? 'bg-gray-400/20 text-gray-300 border border-gray-400/30' :
                    index === 2 ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                    'bg-gray-800/60 text-gray-500 border border-gray-700/50'
                  }`}>
                    #{index + 1}
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white capitalize">{model.model.replace(/_/g, ' ')}</h3>
                    {model.is_best && (
                      <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded-full border border-indigo-500/30 inline-flex items-center gap-1 mt-1">
                        <Award className="w-2.5 h-2.5" /> Best Model
                      </span>
                    )}
                  </div>
                </div>
                {expandedModel === model.model
                  ? <ChevronUp className="w-4 h-4 text-gray-400" />
                  : <ChevronDown className="w-4 h-4 text-gray-400" />
                }
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-gray-800/40 rounded-lg p-2.5 text-center border border-gray-700/30">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">F1 Score</p>
                  <p className="text-sm font-bold text-emerald-400 mt-0.5">{(model.f1 * 100).toFixed(2)}%</p>
                </div>
                <div className="bg-gray-800/40 rounded-lg p-2.5 text-center border border-gray-700/30">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">ROC AUC</p>
                  <p className="text-sm font-bold text-purple-400 mt-0.5">{(model.roc_auc * 100).toFixed(2)}%</p>
                </div>
              </div>
              {expandedModel === model.model && (
                <div className="mt-4 pt-4 border-t border-gray-800/50 space-y-2.5 text-xs animate-fade-in">
                  {[
                    { label: 'Precision', value: `${(model.precision * 100).toFixed(2)}%`, color: 'text-indigo-400' },
                    { label: 'Recall', value: `${(model.recall * 100).toFixed(2)}%`, color: 'text-pink-400' },
                    { label: 'CV Mean', value: `${(model.cv_mean * 100).toFixed(2)}%`, color: 'text-cyan-400' },
                    { label: 'CV Std', value: `${(model.cv_std * 100).toFixed(2)}%`, color: 'text-gray-300' },
                  ].map((metric) => (
                    <div key={metric.label} className="flex justify-between items-center">
                      <span className="text-gray-400">{metric.label}</span>
                      <span className={`font-mono font-semibold ${metric.color}`}>{metric.value}</span>
                    </div>
                  ))}
                  <div className="flex justify-between items-center pt-1 border-t border-gray-800/40">
                    <span className="text-gray-400 flex items-center gap-1"><Clock className="w-3 h-3" /> Training Time</span>
                    <span className="font-mono text-gray-300">{model.training_time.toFixed(1)}s</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Hyperparameter Tuning */}
      <div className="glass-card p-6">
        <SectionHeader icon={Zap} title="Hyperparameter Tuning" subtitle="Optimized parameters for best model" />
        <div className="bg-gray-800/30 rounded-xl p-5 border border-gray-700/30">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-4 h-4 text-indigo-400" />
            <span className="text-sm font-semibold text-white capitalize">
              {summary.hyperparameter_tuning.model?.replace(/_/g, ' ')}
            </span>
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2.5 py-0.5 rounded-full border border-emerald-500/30">
              CV Score: {(summary.hyperparameter_tuning.best_cv_score * 100).toFixed(2)}%
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(summary.hyperparameter_tuning.best_params || {}).map(([key, value]) => (
              <div key={key} className="bg-gray-800/60 rounded-lg p-3 hover:bg-gray-800/80 transition-all duration-200 border border-gray-700/20 hover:border-gray-600/40">
                <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{key}</p>
                <p className="text-sm text-white font-mono font-semibold">{String(value)}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
