import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { LoadingScreen } from './components/LoadingScreen';
import { ErrorScreen } from './components/ErrorScreen';
import { OverviewTab } from './components/OverviewTab';
import { ModelsTab } from './components/ModelsTab';
import { AnalysisTab } from './components/AnalysisTab';
import { PredictTab } from './components/PredictTab';
import { PlotsTab } from './components/PlotsTab';
import type { SummaryData, FeatureData, PlotInfo, PredictionResult } from './types';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

type TabId = 'overview' | 'models' | 'analysis' | 'predict' | 'plots';

function App() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [features, setFeatures] = useState<FeatureData | null>(null);
  const [plots, setPlots] = useState<PlotInfo[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
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
      if (plotsRes.ok) {
        const plotData = await plotsRes.json();
        setPlots(plotData.plots || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to API');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handlePredict = async (data: Record<string, unknown>) => {
    setPrediction(null);
    setPredLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (res.ok) setPrediction(await res.json());
      else {
        const errData = await res.json();
        alert(`Prediction failed: ${errData.detail}`);
      }
    } catch {
      alert('Failed to connect to API');
    } finally {
      setPredLoading(false);
    }
  };

  if (loading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} onRetry={fetchData} />;

  return (
    <div className="min-h-screen flex flex-col">
      <Header activeTab={activeTab} onTabChange={setActiveTab} onRefresh={fetchData} />

      <main className="max-w-7xl mx-auto px-4 py-6 w-full flex-1">
        {activeTab === 'overview' && summary && (
          <OverviewTab summary={summary} />
        )}

        {activeTab === 'models' && summary && (
          <ModelsTab
            summary={summary}
            expandedModel={expandedModel}
            onToggleModel={(model) => setExpandedModel(expandedModel === model ? null : model)}
          />
        )}

        {activeTab === 'analysis' && (
          <AnalysisTab summary={summary} features={features} />
        )}

        {activeTab === 'predict' && (
          <PredictTab
            prediction={prediction}
            predLoading={predLoading}
            onPredict={handlePredict}
          />
        )}

        {activeTab === 'plots' && (
          <PlotsTab plots={plots} apiUrl={API_URL} />
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;
