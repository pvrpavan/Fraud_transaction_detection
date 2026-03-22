import { Logo } from './Logo';

export function Footer() {
  return (
    <footer className="border-t border-gray-800/50 mt-12">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Logo className="w-6 h-6" />
            <div>
              <span className="text-sm font-semibold text-gray-300">
                Fraud<span className="gradient-text">Guard</span> AI
              </span>
              <span className="text-xs text-gray-600 ml-2">v2.0.0</span>
            </div>
          </div>
          <div className="flex items-center gap-6 text-xs text-gray-500">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
              Machine Learning Powered
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-500" />
              FastAPI + React
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-pink-500" />
              Final Year Project
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
