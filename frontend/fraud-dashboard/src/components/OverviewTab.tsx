import {
  Target, Shield, Zap, AlertTriangle, Brain, Database,
  GitBranch, Layers, CheckCircle, PieChart as PieChartIcon
} from 'lucide-react';
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend
} from 'recharts';
import { StatCard } from './StatCard';
import { MetricBadge } from './MetricBadge';
import { SectionHeader } from './SectionHeader';
import { ConfusionMatrix } from './ConfusionMatrix';
import type { SummaryData } from '../types';

interface OverviewTabProps {
  summary: SummaryData;
}

export function OverviewTab({ summary }: OverviewTabProps) {
  const pieData = [
    { name: 'True Negative', value: summary.confusion_matrix.true_negatives, color: '#10B981' },
    { name: 'True Positive', value: summary.confusion_matrix.true_positives, color: '#6366F1' },
    { name: 'False Positive', value: summary.confusion_matrix.false_positives, color: '#F59E0B' },
    { name: 'False Negative', value: summary.confusion_matrix.false_negatives, color: '#EF4444' },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={Target} label="Accuracy" value={`${(summary.best_model.accuracy * 100).toFixed(2)}%`} subvalue="Overall correctness" color="text-emerald-400" glow="glow-emerald" />
        <StatCard icon={Shield} label="Fraud Recall" value={`${(summary.best_model.recall * 100).toFixed(2)}%`} subvalue="Fraud detection rate" color="text-indigo-400" glow="glow-indigo" />
        <StatCard icon={Zap} label="ROC-AUC" value={summary.best_model.roc_auc.toFixed(4)} subvalue="Discrimination power" color="text-purple-400" />
        <StatCard icon={AlertTriangle} label="F1 Score" value={`${(summary.best_model.f1_score * 100).toFixed(2)}%`} subvalue="Balanced metric" color="text-amber-400" />
      </div>

      {/* Best Model & Dataset */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="glass-card p-5 glow-indigo">
          <SectionHeader icon={Brain} title="Best Model" subtitle="Auto-selected optimal performer" />
          <div className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-400">Selected Model</p>
            <p className="text-xl font-bold gradient-text">{summary.best_model.name}</p>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <MetricBadge label="Precision" value={summary.best_model.precision} good={summary.best_model.precision > 0.9} />
            <MetricBadge label="Specificity" value={summary.best_model.specificity} good={summary.best_model.specificity > 0.9} />
            <MetricBadge label="MCC" value={summary.best_model.mcc} good={summary.best_model.mcc > 0.7} />
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs text-gray-500">
            <Zap className="w-3 h-3" />
            <span>Pipeline completed in {summary.pipeline_time.toFixed(1)}s</span>
          </div>
        </div>

        <div className="glass-card p-5">
          <SectionHeader icon={Database} title="Dataset Overview" subtitle="Transaction analysis" />
          <div className="space-y-3">
            {[
              { label: 'Total Transactions', value: summary.dataset.total_transactions.toLocaleString(), color: 'text-white' },
              { label: 'Fraud Transactions', value: summary.dataset.fraud_transactions.toLocaleString(), color: 'text-rose-400' },
              { label: 'Legitimate Transactions', value: summary.dataset.legitimate_transactions.toLocaleString(), color: 'text-emerald-400' },
              { label: 'Fraud Ratio', value: `${(summary.dataset.fraud_ratio * 100).toFixed(4)}%`, color: 'text-amber-400' },
              { label: 'Sampling Strategy', value: summary.sampling.strategy, color: 'text-indigo-400' },
              { label: 'Training Samples', value: summary.sampling.sampled_rows.toLocaleString(), color: 'text-white' },
              { label: 'Imbalance Method', value: summary.imbalance_handling.method?.toUpperCase(), color: 'text-purple-400' },
            ].map((item) => (
              <div key={item.label} className="flex justify-between items-center py-2 border-b border-gray-800/60 last:border-0 hover:bg-gray-800/20 px-2 -mx-2 rounded transition-colors">
                <span className="text-sm text-gray-400">{item.label}</span>
                <span className={`text-sm font-bold ${item.color}`}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Confusion Matrix & Pie Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="glass-card p-5">
          <SectionHeader icon={Layers} title="Confusion Matrix" subtitle="Classification results breakdown" />
          <ConfusionMatrix cm={summary.confusion_matrix} />
        </div>
        <div className="glass-card p-5">
          <SectionHeader icon={PieChartIcon} title="Prediction Distribution" />
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={95} paddingAngle={3} dataKey="value">
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }}
                formatter={(value: number) => [value.toLocaleString(), '']}
              />
              <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Pipeline Summary Bar */}
      <div className="glass-card p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-indigo-400" />
              <span className="text-gray-400">Features:</span>
              <span className="text-white font-medium">{summary.preprocessing.n_features}</span>
            </div>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="text-gray-400">Models Tested:</span>
              <span className="text-white font-medium">{summary.model_comparison.length}</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              <span className="text-gray-400">Tuned:</span>
              <span className="text-white font-medium">{summary.hyperparameter_tuning.model}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span className="text-sm text-emerald-400 font-medium">Pipeline Complete</span>
          </div>
        </div>
      </div>
    </div>
  );
}
