import { Logo } from './Logo';
import { Heart, Github, Cpu, Code2 } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-gray-800/50 mt-12 bg-gray-950/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <Logo className="w-6 h-6" />
            <div>
              <span className="text-sm font-semibold text-gray-300">
                Fraud<span className="gradient-text">Guard</span> AI
              </span>
              <span className="text-[10px] text-gray-600 ml-2 bg-gray-800/60 px-1.5 py-0.5 rounded">v2.0.0</span>
            </div>
          </div>
          <div className="flex items-center gap-5 text-xs text-gray-500">
            <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-gray-800/30 border border-gray-800/50">
              <Cpu className="w-3 h-3 text-indigo-400" />
              ML Powered
            </span>
            <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-gray-800/30 border border-gray-800/50">
              <Code2 className="w-3 h-3 text-purple-400" />
              FastAPI + React
            </span>
            <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-gray-800/30 border border-gray-800/50">
              <Heart className="w-3 h-3 text-pink-400" />
              Final Year Project
            </span>
            <a
              href="https://github.com/pvrpavan/Fraud_transaction_detection"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-gray-800/30 border border-gray-800/50 hover:border-gray-600 hover:text-gray-300 transition-all"
            >
              <Github className="w-3 h-3" />
              GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
