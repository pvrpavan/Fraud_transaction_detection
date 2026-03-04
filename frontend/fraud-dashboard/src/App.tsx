import { useState, useEffect, useCallback } from 'react';
import {
  Shield, AlertTriangle, CheckCircle, TrendingUp, Database,
  BarChart3, Brain, Zap, Activity, Search, RefreshCw,
  ChevronDown, ChevronUp, Eye, Cpu, FileBarChart, GitBranch,
  Layers, Target, PieChart as PieChartIcon
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SummaryData {
  dataset: { total_transactions: number; fraud_transactions: number; legitimate_transactions: number; fraud_ratio: number };
  sampling: { strategy: string; original_rows: number; sampled_rows: number; fraud_count: number; fraud_ratio: number };
  preprocessing: { n_features: number; feature_names: string[] };
  imbalance_handling: { method: string; original_train_size: number; balanced_train_size: number; original_fraud_ratio: number; balanced_fraud_ratio: number };
  model_comparison: ModelResult[];
  best_model: { name: string; accuracy: number; precision: number; recall: number; f1_score: number; roc_auc: number; specificity: number; mcc: number };
  confusion_matrix: { true_negatives: number; false_positives: number; false_negatives: number; true_positives: number };
  hyperparameter_tuning: { model: string; best_params: Record<string, unknown>; best_cv_score: number };
  pipeline_time: number;
}

interface ModelResult {
  model: string; precision: number; recall: number; f1: number; roc_auc: number;
  cv_mean: number; cv_std: number; training_time: number; is_best: boolean;
}

interface FeatureData { feature_importance: Record<string, number>; top_features: string[] }
interface PlotInfo { filename: string; url: string; size: number }
interface PredictionResult { is_fraud: boolean; fraud_probability: number; risk_level: string; confidence: number }

function StatCard({ icon: Icon, label, value, subvalue, color, glow }: {
  icon: React.ElementType; label: string; value: string; subvalue?: string; color: string; glow?: string;
}) {
  return (
    <div className={`glass-card p-5 ${glow || ''} animate-slide-up hover:scale-[1.02] transition-transform`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">{label}</p>
          <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
          {subvalue && <p className="text-xs text-gray-500 mt-1">{subvalue}</p>}
        </div>
        <div className="p-2.5 rounded-lg bg-gray-800/60"><Icon className={`w-5 h-5 ${color}`} /></div>
      </div>
    </div>
  );
}

function MetricBadge({ label, value, good }: { label: string; value: number; good?: boolean }) {
  const color = good === undefined ? 'text-gray-300' : good ? 'text-emerald-400' : 'text-rose-400';
  return (
    <div className="flex flex-col items-center p-3 rounded-lg bg-gray-800/40">
      <span className="text-xs text-gray-400 mb-1">{label}</span>
      <span className={`text-lg font-bold ${color}`}>{(value * 100).toFixed(2)}%</span>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, subtitle }: { icon: React.ElementType; title: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20"><Icon className="w-5 h-5 text-indigo-400" /></div>
      <div>
        <h2 className="text-lg font-bold text-white">{title}</h2>
        {subtitle && <p className="text-xs text-gray-400">{subtitle}</p>}
      </div>
    </div>
  );
}

function ConfusionMatrixVisual({ cm }: { cm: { true_negatives: number; false_positives: number; false_negatives: number; true_positives: number } }) {
  const total = cm.true_negatives + cm.false_positives + cm.false_negatives + cm.true_positives;
  const cells = [
    { label: 'True Negative', value: cm.true_negatives, color: 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' },
    { label: 'False Positive', value: cm.false_positives, color: 'bg-amber-500/20 border-amber-500/30 text-amber-400' },
    { label: 'False Negative', value: cm.false_negatives, color: 'bg-rose-500/20 border-rose-500/30 text-rose-400' },
    { label: 'True Positive', value: cm.true_positives, color: 'bg-indigo-500/20 border-indigo-500/30 text-indigo-400' },
  ];
  return (
    <div className="grid grid-cols-2 gap-3">
      {cells.map((cell) => (
        <div key={cell.label} className={`${cell.color} border rounded-lg p-4 text-center`}>
          <p className="text-xs text-gray-400 mb-1">{cell.label}</p>
          <p className="text-xl font-bold">{cell.value.toLocaleString()}</p>
          <p className="text-xs text-gray-500 mt-1">{(cell.value / total * 100).toFixed(1)}%</p>
        </div>
      ))}
    </div>
  );
}

function PredictionForm({ onPredict, loading }: { onPredict: (data: Record<string, unknown>) => void; loading: boolean }) {
  const [formData, setFormData] = useState({
    step: '1', type: 'TRANSFER', amount: '181000', oldbalanceOrg: '181000',
    newbalanceOrig: '0', oldbalanceDest: '0', newbalanceDest: '0',
  });

  const fields = [
    { key: 'step', label: 'Step (Time)', type: 'number' },
    { key: 'type', label: 'Transaction Type', type: 'select', options: ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'] },
    { key: 'amount', label: 'Amount', type: 'number' },
    { key: 'oldbalanceOrg', label: 'Old Balance (Origin)', type: 'number' },
    { key: 'newbalanceOrig', label: 'New Balance (Origin)', type: 'number' },
    { key: 'oldbalanceDest', label: 'Old Balance (Dest)', type: 'number' },
    { key: 'newbalanceDest', label: 'New Balance (Dest)', type: 'number' },
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict({
      step: parseInt(formData.step), type: formData.type, amount: parseFloat(formData.amount),
      oldbalanceOrg: parseFloat(formData.oldbalanceOrg), newbalanceOrig: parseFloat(formData.newbalanceOrig),
      oldbalanceDest: parseFloat(formData.oldbalanceDest), newbalanceDest: parseFloat(formData.newbalanceDest),
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      {fields.map((field) => (
        <div key={field.key}>
          <label className="text-xs text-gray-400 mb-1 block">{field.label}</label>
          {field.type === 'select' ? (
            <select value={formData[field.key as keyof typeof formData]}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              className="w-full bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors">
              {field.options?.map((opt) => (<option key={opt} value={opt}>{opt}</option>))}
            </select>
          ) : (
            <input type="number" step="any" value={formData[field.key as keyof typeof formData]}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              className="w-full bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors" />
          )}
        </div>
      ))}
      <button type="submit" disabled={loading}
        className="w-full py-2.5 px-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg text-sm font-semibold text-white transition-all disabled:opacity-50 flex items-center justify-center gap-2">
        {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
        {loading ? 'Analyzing...' : 'Detect Fraud'}
      </button>
    </form>
  );
}

function App() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [features, setFeatures] = useState<FeatureData | null>(null);
  const [plots, setPlots] = useState<PlotInfo[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'models' | 'analysis' | 'predict' | 'plots'>('overview');
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [summaryRes, featuresRes, plotsRes] = await Promise.all([
        fetch(`${API_URL}/api/results/summary`),
        fetch(`${API_URL}/api/results/feature-importance`),
        fetch(`${API_URL}/api/plots`),
      ]);
      if (summaryRes.ok) setSummary(await summaryRes.json());
      else throw new Error('Failed to load results. Ensure the ML pipeline has been run.');
      if (featuresRes.ok) setFeatures(await featuresRes.json());
      if (plotsRes.ok) { const plotData = await plotsRes.json(); setPlots(plotData.plots || []); }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to API');
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handlePredict = async (data: Record<string, unknown>) => {
    setPredLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
      });
      if (res.ok) setPrediction(await res.json());
      else { const errData = await res.json(); alert(`Prediction failed: ${errData.detail}`); }
    } catch { alert('Failed to connect to API'); }
    finally { setPredLoading(false); }
  };

  const modelComparisonData = summary?.model_comparison?.map((m) => ({
    name: m.model.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    Precision: parseFloat((m.precision * 100).toFixed(2)),
    Recall: parseFloat((m.recall * 100).toFixed(2)),
    'F1 Score': parseFloat((m.f1 * 100).toFixed(2)),
    'ROC AUC': parseFloat((m.roc_auc * 100).toFixed(2)),
  })) || [];

  const featureData = features?.feature_importance
    ? Object.entries(features.feature_importance).slice(0, 12).map(([name, value]) => ({
        name: name.length > 18 ? name.substring(0, 18) + '...' : name,
        fullName: name,
        importance: parseFloat((value as number).toFixed(4)),
      }))
    : [];

  const pieData = summary?.confusion_matrix ? [
    { name: 'True Negative', value: summary.confusion_matrix.true_negatives, color: '#10B981' },
    { name: 'True Positive', value: summary.confusion_matrix.true_positives, color: '#6366F1' },
    { name: 'False Positive', value: summary.confusion_matrix.false_positives, color: '#F59E0B' },
    { name: 'False Negative', value: summary.confusion_matrix.false_negatives, color: '#EF4444' },
  ] : [];

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: Activity },
    { id: 'models' as const, label: 'Models', icon: Cpu },
    { id: 'analysis' as const, label: 'Analysis', icon: FileBarChart },
    { id: 'predict' as const, label: 'Predict', icon: Search },
    { id: 'plots' as const, label: 'Plots', icon: PieChartIcon },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="relative mb-6">
            <Shield className="w-16 h-16 text-indigo-400 mx-auto animate-pulse" />
            <div className="absolute inset-0 w-16 h-16 mx-auto rounded-full bg-indigo-500/20 animate-ping" />
          </div>
          <p className="text-lg text-gray-300">Loading Fraud Detection Dashboard...</p>
          <p className="text-sm text-gray-500 mt-2">Connecting to ML backend</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glass-card p-8 max-w-lg text-center">
          <AlertTriangle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
          <p className="text-gray-400 mb-4">{error}</p>
          <div className="bg-gray-800/40 rounded-lg p-4 text-left text-sm text-gray-300 mb-4">
            <p className="font-semibold mb-2">Quick Start:</p>
            <ol className="list-decimal list-inside space-y-1 text-gray-400">
              <li>Run the ML pipeline: <code className="text-indigo-400">python run.py --data data/your_data.csv</code></li>
              <li>Start the API: <code className="text-indigo-400">uvicorn backend.app.main:app --reload</code></li>
              <li>Refresh this page</li>
            </ol>
          </div>
          <button onClick={fetchData} className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 mx-auto">
            <RefreshCw className="w-4 h-4" /> Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 border-b border-gray-800/50 bg-gray-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/20">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">FraudGuard AI</h1>
              <p className="text-xs text-gray-400">Intelligent Transaction Monitoring</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-emerald-400 font-medium">System Active</span>
            </div>
            <button onClick={fetchData} className="p-2 rounded-lg bg-gray-800/60 border border-gray-700 hover:border-gray-600 transition-colors" title="Refresh Data">
              <RefreshCw className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-4">
          <nav className="flex gap-1 overflow-x-auto pb-0">
            {tabs.map((tab) => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-all border-b-2 whitespace-nowrap ${activeTab === tab.id ? 'text-indigo-400 border-indigo-400' : 'text-gray-400 border-transparent hover:text-gray-300 hover:border-gray-600'}`}>
                <tab.icon className="w-4 h-4" />{tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'overview' && summary && (
          <div className="space-y-6 animate-fade-in">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard icon={Target} label="Accuracy" value={`${(summary.best_model.accuracy * 100).toFixed(2)}%`} subvalue="Overall correctness" color="text-emerald-400" glow="glow-emerald" />
              <StatCard icon={Shield} label="Fraud Recall" value={`${(summary.best_model.recall * 100).toFixed(2)}%`} subvalue="Fraud detection rate" color="text-indigo-400" glow="glow-indigo" />
              <StatCard icon={Zap} label="ROC-AUC" value={summary.best_model.roc_auc.toFixed(4)} subvalue="Discrimination power" color="text-purple-400" />
              <StatCard icon={AlertTriangle} label="F1 Score" value={`${(summary.best_model.f1_score * 100).toFixed(2)}%`} subvalue="Balanced metric" color="text-amber-400" />
            </div>

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
                    <div key={item.label} className="flex justify-between items-center py-2 border-b border-gray-800 last:border-0">
                      <span className="text-sm text-gray-400">{item.label}</span>
                      <span className={`text-sm font-bold ${item.color}`}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="glass-card p-5">
                <SectionHeader icon={Layers} title="Confusion Matrix" subtitle="Classification results breakdown" />
                <ConfusionMatrixVisual cm={summary.confusion_matrix} />
              </div>
              <div className="glass-card p-5">
                <SectionHeader icon={PieChartIcon} title="Prediction Distribution" />
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={95} paddingAngle={3} dataKey="value">
                      {pieData.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.color} stroke="transparent" />))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} formatter={(value: number) => [value.toLocaleString(), '']} />
                    <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-card p-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-6 text-sm">
                  <div className="flex items-center gap-2"><GitBranch className="w-4 h-4 text-indigo-400" /><span className="text-gray-400">Features:</span><span className="text-white font-medium">{summary.preprocessing.n_features}</span></div>
                  <div className="flex items-center gap-2"><Layers className="w-4 h-4 text-purple-400" /><span className="text-gray-400">Models Tested:</span><span className="text-white font-medium">{summary.model_comparison.length}</span></div>
                  <div className="flex items-center gap-2"><Zap className="w-4 h-4 text-amber-400" /><span className="text-gray-400">Tuned:</span><span className="text-white font-medium">{summary.hyperparameter_tuning.model}</span></div>
                </div>
                <div className="flex items-center gap-2"><CheckCircle className="w-4 h-4 text-emerald-400" /><span className="text-sm text-emerald-400 font-medium">Pipeline Complete</span></div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && summary && (
          <div className="space-y-6 animate-fade-in">
            <div className="glass-card p-5">
              <SectionHeader icon={BarChart3} title="Model Performance Comparison" subtitle="Side-by-side metric comparison" />
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={modelComparisonData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-30} textAnchor="end" height={80} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} formatter={(value: number) => [`${value.toFixed(2)}%`, '']} />
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
                <div key={model.model}
                  className={`glass-card p-4 cursor-pointer transition-all hover:scale-[1.02] ${model.is_best ? 'ring-2 ring-indigo-500/50 glow-indigo' : ''}`}
                  onClick={() => setExpandedModel(expandedModel === model.model ? null : model.model)}>
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="text-sm font-bold text-white capitalize">{model.model.replace(/_/g, ' ')}</h3>
                      {model.is_best && <span className="text-xs bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded-full border border-indigo-500/30">Best Model</span>}
                    </div>
                    {expandedModel === model.model ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-gray-800/40 rounded-lg p-2 text-center"><p className="text-xs text-gray-500">F1</p><p className="text-sm font-bold text-emerald-400">{(model.f1 * 100).toFixed(2)}%</p></div>
                    <div className="bg-gray-800/40 rounded-lg p-2 text-center"><p className="text-xs text-gray-500">AUC</p><p className="text-sm font-bold text-purple-400">{(model.roc_auc * 100).toFixed(2)}%</p></div>
                  </div>
                  {expandedModel === model.model && (
                    <div className="mt-3 pt-3 border-t border-gray-800 space-y-2 text-xs">
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
                  <span className="text-sm font-semibold text-white capitalize">{summary.hyperparameter_tuning.model?.replace(/_/g, ' ')}</span>
                  <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">CV Score: {(summary.hyperparameter_tuning.best_cv_score * 100).toFixed(2)}%</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {Object.entries(summary.hyperparameter_tuning.best_params || {}).map(([key, value]) => (
                    <div key={key} className="bg-gray-800/60 rounded-lg p-2"><p className="text-xs text-gray-500">{key}</p><p className="text-sm text-white font-mono">{String(value)}</p></div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-6 animate-fade-in">
            {featureData.length > 0 && (
              <div className="glass-card p-5">
                <SectionHeader icon={TrendingUp} title="Feature Importance" subtitle="Most influential features for fraud detection" />
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={featureData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={100} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0' }} />
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
                  <div>
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">Before Balancing</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm"><span className="text-gray-400">Training Samples</span><span className="text-white">{summary.imbalance_handling.original_train_size?.toLocaleString()}</span></div>
                      <div className="flex justify-between text-sm"><span className="text-gray-400">Fraud Ratio</span><span className="text-rose-400">{(summary.imbalance_handling.original_fraud_ratio * 100).toFixed(4)}%</span></div>
                      <div className="w-full bg-gray-800 rounded-full h-3 mt-2"><div className="bg-rose-500 h-3 rounded-full" style={{ width: `${Math.max(summary.imbalance_handling.original_fraud_ratio * 100 * 50, 1)}%` }} /></div>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">After Balancing ({summary.imbalance_handling.method?.toUpperCase()})</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm"><span className="text-gray-400">Training Samples</span><span className="text-white">{summary.imbalance_handling.balanced_train_size?.toLocaleString()}</span></div>
                      <div className="flex justify-between text-sm"><span className="text-gray-400">Fraud Ratio</span><span className="text-emerald-400">{(summary.imbalance_handling.balanced_fraud_ratio * 100).toFixed(2)}%</span></div>
                      <div className="w-full bg-gray-800 rounded-full h-3 mt-2"><div className="bg-emerald-500 h-3 rounded-full" style={{ width: `${summary.imbalance_handling.balanced_fraud_ratio * 100}%` }} /></div>
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
                    <span key={name} className="text-xs bg-gray-800/60 text-gray-300 px-3 py-1.5 rounded-full border border-gray-700/50 hover:border-indigo-500/30 transition-colors">{name}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
            <div className="glass-card p-5">
              <SectionHeader icon={Search} title="Transaction Analyzer" subtitle="Enter transaction details to check for fraud" />
              <PredictionForm onPredict={handlePredict} loading={predLoading} />
            </div>
            <div className="glass-card p-5">
              <SectionHeader icon={Eye} title="Analysis Result" subtitle="Real-time fraud probability assessment" />
              {prediction ? (
                <div className="space-y-4">
                  <div className={`rounded-lg p-6 text-center border ${prediction.is_fraud ? 'bg-rose-500/10 border-rose-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}>
                    {prediction.is_fraud ? <AlertTriangle className="w-12 h-12 text-rose-400 mx-auto mb-2" /> : <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-2" />}
                    <p className={`text-2xl font-bold ${prediction.is_fraud ? 'text-rose-400' : 'text-emerald-400'}`}>{prediction.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}</p>
                    <p className="text-sm text-gray-400 mt-1">Risk Level: <span className={`font-bold ${prediction.risk_level === 'HIGH' ? 'text-rose-400' : prediction.risk_level === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400'}`}>{prediction.risk_level}</span></p>
                  </div>
                  <div className="bg-gray-800/40 rounded-lg p-4">
                    <p className="text-xs text-gray-400 mb-2">Fraud Probability</p>
                    <div className="w-full bg-gray-700 rounded-full h-4 relative">
                      <div className={`h-4 rounded-full transition-all duration-500 ${prediction.fraud_probability > 0.7 ? 'bg-rose-500' : prediction.fraud_probability > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`} style={{ width: `${prediction.fraud_probability * 100}%` }} />
                      <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-white">{(prediction.fraud_probability * 100).toFixed(2)}%</span>
                    </div>
                  </div>
                  <div className="bg-gray-800/40 rounded-lg p-4">
                    <p className="text-xs text-gray-400 mb-1">Model Confidence</p>
                    <p className="text-lg font-bold text-white">{(prediction.confidence * 100).toFixed(2)}%</p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                  <Search className="w-12 h-12 mb-3 opacity-30" />
                  <p className="text-sm">Enter transaction details and click "Detect Fraud"</p>
                  <p className="text-xs mt-1">Results will appear here</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'plots' && (
          <div className="space-y-6 animate-fade-in">
            <SectionHeader icon={PieChartIcon} title="Generated Visualizations" subtitle="ML pipeline output plots" />
            {plots.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {plots.map((plot) => (
                  <div key={plot.filename} className="glass-card p-4">
                    <p className="text-sm font-medium text-gray-300 mb-3 capitalize">{plot.filename.replace('.png', '').replace(/_/g, ' ')}</p>
                    <img src={`${API_URL}${plot.url}`} alt={plot.filename} className="w-full rounded-lg" loading="lazy" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="glass-card p-8 text-center">
                <PieChartIcon className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-400">No plots generated yet</p>
                <p className="text-sm text-gray-500 mt-1">Run the ML pipeline to generate visualizations</p>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-gray-800/50 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-3 text-xs text-gray-500">
          <div className="flex items-center gap-2"><Shield className="w-4 h-4 text-indigo-500" /><span>FraudGuard AI - Intelligent Fraud Detection System v1.0.0</span></div>
          <div className="flex items-center gap-4"><span>Powered by Machine Learning</span><span>Built with FastAPI + React</span></div>
        </div>
      </footer>
    </div>
  );
}

export default App;
