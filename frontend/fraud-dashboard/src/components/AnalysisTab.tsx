import { TrendingUp, Activity, GitBranch } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { SectionHeader } from './SectionHeader';
import type { SummaryData, FeatureData } from '../types';

interface AnalysisTabProps {
  summary: SummaryData | null;
  features: FeatureData | null;
}

export function AnalysisTab({ summary, features }: AnalysisTabProps) {
  const featureData = features?.feature_importance
    ? Object.entries(features.feature_importance).slice(0, 12).map(([name, value]) => ({
        name: name.length > 18 ? name.substring(0, 18) + '...' : name,
        fullName: name,
        importance: parseFloat((value as number).toFixed(4)),
      }))
    : [];

  return (
    <div className="space-y-6 animate-fade-in">
      {featureData.length > 0 && (
        <div className="glass-card p-5">
          <SectionHeader icon={TrendingUp} title="Feature Importance" subtitle="Most influential features for fraud detection" />
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={featureData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={100} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {featureData.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={`hsl(${240 - index * 15}, 70%, ${55 + index * 2}%)`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {summary && (
        <div className="glass-card p-5">
          <SectionHeader icon={Activity} title="Class Imbalance Analysis" subtitle="How the severe class imbalance was handled" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="p-4 rounded-lg bg-gray-800/20 border border-gray-800/50">
              <h4 className="text-sm font-semibold text-gray-300 mb-3">Before Balancing</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Training Samples</span>
                  <span className="text-white">{summary.imbalance_handling.original_train_size?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Fraud Ratio</span>
                  <span className="text-rose-400">{(summary.imbalance_handling.original_fraud_ratio * 100).toFixed(4)}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3 mt-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-rose-600 to-rose-400 h-3 rounded-full transition-all duration-700"
                    style={{ width: `${Math.max(summary.imbalance_handling.original_fraud_ratio * 100 * 50, 1)}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="p-4 rounded-lg bg-gray-800/20 border border-gray-800/50">
              <h4 className="text-sm font-semibold text-gray-300 mb-3">
                After Balancing ({summary.imbalance_handling.method?.toUpperCase()})
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Training Samples</span>
                  <span className="text-white">{summary.imbalance_handling.balanced_train_size?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Fraud Ratio</span>
                  <span className="text-emerald-400">{(summary.imbalance_handling.balanced_fraud_ratio * 100).toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3 mt-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-3 rounded-full transition-all duration-700"
                    style={{ width: `${summary.imbalance_handling.balanced_fraud_ratio * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {summary && (
        <div className="glass-card p-5">
          <SectionHeader icon={GitBranch} title="Engineered Features" subtitle={`${summary.preprocessing.n_features} features used`} />
          <div className="flex flex-wrap gap-2">
            {summary.preprocessing.feature_names.map((name) => (
              <span
                key={name}
                className="text-xs bg-gray-800/60 text-gray-300 px-3 py-1.5 rounded-full border border-gray-700/50 hover:border-indigo-500/30 hover:bg-indigo-500/5 hover:text-indigo-300 transition-all duration-200 cursor-default"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
