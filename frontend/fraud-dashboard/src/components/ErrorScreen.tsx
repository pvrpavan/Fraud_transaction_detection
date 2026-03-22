import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Logo } from './Logo';

interface ErrorScreenProps {
  error: string;
  onRetry: () => void;
}

export function ErrorScreen({ error, onRetry }: ErrorScreenProps) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="glass-card p-8 max-w-lg w-full text-center">
        <div className="inline-flex p-3 rounded-full bg-amber-500/10 mb-4">
          <AlertTriangle className="w-12 h-12 text-amber-400" />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
        <p className="text-gray-400 mb-6">{error}</p>

        <div className="bg-gray-800/40 rounded-xl p-5 text-left text-sm text-gray-300 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Logo className="w-5 h-5" />
            <p className="font-semibold text-white">Quick Start Guide</p>
          </div>
          <ol className="list-decimal list-inside space-y-2 text-gray-400">
            <li>
              Run the ML pipeline:
              <code className="ml-1 text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded text-xs">
                python run.py --data data/your_data.csv
              </code>
            </li>
            <li>
              Start the API:
              <code className="ml-1 text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded text-xs">
                uvicorn backend.app.main:app --reload
              </code>
            </li>
            <li>Refresh this page</li>
          </ol>
        </div>

        <button
          onClick={onRetry}
          className="px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg text-sm font-semibold transition-all duration-300 flex items-center gap-2 mx-auto shadow-lg shadow-indigo-500/20"
        >
          <RefreshCw className="w-4 h-4" /> Retry Connection
        </button>
      </div>
    </div>
  );
}
