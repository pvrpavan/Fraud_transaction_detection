import { RefreshCw, Activity, Cpu, FileBarChart, Search, PieChart as PieChartIcon } from 'lucide-react';
import { LogoFull } from './Logo';

type TabId = 'overview' | 'models' | 'analysis' | 'predict' | 'plots';

interface HeaderProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  onRefresh: () => void;
}

const tabs: { id: TabId; label: string; icon: React.ElementType; description: string }[] = [
  { id: 'overview', label: 'Overview', icon: Activity, description: 'Dashboard' },
  { id: 'models', label: 'Models', icon: Cpu, description: 'Compare' },
  { id: 'analysis', label: 'Analysis', icon: FileBarChart, description: 'Features' },
  { id: 'predict', label: 'Predict', icon: Search, description: 'Test' },
  { id: 'plots', label: 'Plots', icon: PieChartIcon, description: 'Visuals' },
];

export function Header({ activeTab, onTabChange, onRefresh }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 border-b border-gray-800/50 bg-gray-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
        <LogoFull />
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-emerald-400 font-medium">System Active</span>
          </div>
          <button
            onClick={onRefresh}
            className="group p-2.5 rounded-xl bg-gray-800/60 border border-gray-700 hover:border-indigo-500/50 hover:bg-indigo-500/10 transition-all duration-300"
            title="Refresh Data"
          >
            <RefreshCw className="w-4 h-4 text-gray-400 group-hover:text-indigo-400 transition-colors" />
          </button>
        </div>
      </div>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <nav className="flex gap-1 overflow-x-auto pb-0 scrollbar-hide">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`relative flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all duration-200 border-b-2 whitespace-nowrap ${
                activeTab === tab.id
                  ? 'text-indigo-400 border-indigo-400 bg-indigo-500/5'
                  : 'text-gray-400 border-transparent hover:text-gray-300 hover:border-gray-600 hover:bg-gray-800/30'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
              <span className={`hidden sm:inline text-[10px] px-1.5 py-0.5 rounded-full ${
                activeTab === tab.id
                  ? 'bg-indigo-500/20 text-indigo-300'
                  : 'bg-gray-800/60 text-gray-500'
              }`}>
                {tab.description}
              </span>
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
