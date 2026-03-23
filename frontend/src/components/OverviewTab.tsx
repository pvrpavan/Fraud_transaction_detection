import {
  Target, Shield, Zap, AlertTriangle, Brain, Database,
  GitBranch, Layers, CheckCircle, PieChart as PieChartIcon, Clock, TrendingUp
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
    <div className="space-y-8 animate-fade-in">
      {/* Hero Section */}
      <div className="glass-card p-6 relative overflow-hidden glow-indigo">
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-indigo-500/10 via-purple-500/5 to-transparent rounded-bl-full" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-indigo-500/30">
              <Brain className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Best Model: <span className="gradient-text">{summary.best_model.name}</span></h2>
              <p className="text-xs text-gray-400 flex items-center gap-1.5 mt-0.5">
                <Clock className="w-3 h-3" /> Pipeline completed in {summary.pipeline_time.toFixed(1)}s
                <span className="mx-1 text-gray-600">|</span>
                <TrendingUp className="w-3 h-3" /> Auto-selected from {summary.model_comparison.length} models
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={Target} label="Accuracy" value={`${(summary.best_model.accuracy * 100).toFixed(2)}%`} subvalue="Overall correctness" color="text-emerald-400" glow="glow-emerald" delay={0} />
        <StatCard icon={Shield} label="Fraud Recall" value={`${(summary.best_model.recall * 100).toFixed(2)}%`} subvalue="Fraud detection rate" color="text-indigo-400" glow="glow-indigo" delay={100} />
        <StatCard icon={Zap} label="ROC-AUC" value={summary.best_model.roc_auc.toFixed(4)} subvalue="Discrimination power" color="text-purple-400" delay={200} />
        <StatCard icon={AlertTriangle} label="F1 Score" value={`${(summary.best_model.f1_score * 100).toFixed(2)}%`} subvalue="Balanced metric" color="text-amber-400" delay={300} />
      </div>

      {/* Model Details & Dataset */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="glass-card p-6">
          <SectionHeader icon={Brain} title="Model Metrics" subtitle="Detailed performance breakdown" />
          <div className="grid grid-cols-3 gap-3 mb-5">
            <MetricBadge label="Precision" value={summary.best_model.precision} good={summary.best_model.precision > 0.9} />
            <MetricBadge label="Specificity" value={summary.best_model.specificity} good={summary.best_model.specificity > 0.9} />
            <MetricBadge label="MCC" value={summary.best_model.mcc} good={summary.best_model.mcc > 0.7} />
          </div>
          <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/30">
            <p className="text-xs text-gray-400 mb-2 font-medium uppercase tracking-wider">Hyperparameter Tuning</p>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-semibold text-white capitalize">
                {summary.hyperparameter_tuning.model?.replace(/_/g, ' ')}
              </span>
              <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-500/30">
                CV: {(summary.hyperparameter_tuning.best_cv_score * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

        <div className="glass-card p-6">
          <SectionHeader icon={Database} title="Dataset Overview" subtitle="Transaction analysis" />
          <div className="space-y-2.5">
            {[
              { label: 'Total Transactions', value: summary.dataset.total_transactions.toLocaleString(), color: 'text-white', icon: '~' },
              { label: 'Fraud Transactions', value: summary.dataset.fraud_transactions.toLocaleString(), color: 'text-rose-400', icon: '!' },
              { label: 'Legitimate', value: summary.dataset.legitimate_transactions.toLocaleString(), color: 'text-emerald-400', icon: '+' },
              { label: 'Fraud Ratio', value: `${(summary.dataset.fraud_ratio * 100).toFixed(4)}%`, color: 'text-amber-400', icon: '%' },
              { label: 'Sampling', value: summary.sampling.strategy, color: 'text-indigo-400', icon: '#' },
              { label: 'Training Samples', value: summary.sampling.sampled_rows.toLocaleString(), color: 'text-white', icon: '=' },
              { label: 'Imbalance Method', value: summary.imbalance_handling.method?.toUpperCase(), color: 'text-purple-400', icon: '*' },
            ].map((item) => (
              <div key={item.label} className="flex justify-between items-center py-2 px-3 -mx-1 rounded-lg hover:bg-gray-800/30 transition-all duration-200 group cursor-default">
                <span className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">{item.label}</span>
                <span className={`text-sm font-bold ${item.color} font-mono`}>{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Confusion Matrix & Pie Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="glass-card p-6">
          <SectionHeader icon={Layers} title="Confusion Matrix" subtitle="Classification results breakdown" />
          <ConfusionMatrix cm={summary.confusion_matrix} />
        </div>
        <div className="glass-card p-6">
          <SectionHeader icon={PieChartIcon} title="Prediction Distribution" subtitle="Visual classification summary" />
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={3} dataKey="value" animationDuration={800}>
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} stroke="transparent" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', color: '#e2e8f0', boxShadow: '0 10px 40px rgba(0,0,0,0.3)' }}
                formatter={(value: number) => [value.toLocaleString(), '']}
              />
              <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Pipeline Summary Bar */}
      <div className="glass-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-500/5 border border-indigo-500/10">
              <GitBranch className="w-4 h-4 text-indigo-400" />
              <span className="text-gray-400">Features:</span>
              <span className="text-white font-semibold">{summary.preprocessing.n_features}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-500/5 border border-purple-500/10">
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="text-gray-400">Models:</span>
              <span className="text-white font-semibold">{summary.model_comparison.length}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
              <Zap className="w-4 h-4 text-amber-400" />
              <span className="text-gray-400">Tuned:</span>
              <span className="text-white font-semibold capitalize">{summary.hyperparameter_tuning.model?.replace(/_/g, ' ')}</span>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span className="text-sm text-emerald-400 font-semibold">Pipeline Complete</span>
          </div>
        </div>
      </div>
    </div>
  );
}
