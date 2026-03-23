import { useState } from 'react';
import { PieChart as PieChartIcon, Maximize2, X, Download } from 'lucide-react';
import { SectionHeader } from './SectionHeader';
import type { PlotInfo } from '../types';

interface PlotsTabProps {
  plots: PlotInfo[];
  apiUrl: string;
}

export function PlotsTab({ plots, apiUrl }: PlotsTabProps) {
  const [selectedPlot, setSelectedPlot] = useState<PlotInfo | null>(null);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <SectionHeader icon={PieChartIcon} title="Generated Visualizations" subtitle="ML pipeline output plots" accent="purple" />
        {plots.length > 0 && (
          <span className="text-xs text-gray-500 bg-gray-800/40 px-3 py-1.5 rounded-full border border-gray-700/30">
            {plots.length} plots
          </span>
        )}
      </div>
      {plots.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {plots.map((plot) => (
            <div key={plot.filename} className="glass-card p-5 group hover:border-indigo-500/20 transition-all duration-300 relative">
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm font-semibold text-gray-300 capitalize">
                  {plot.filename.replace('.png', '').replace(/_/g, ' ')}
                </p>
                <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => setSelectedPlot(plot)}
                    className="p-1.5 rounded-lg bg-gray-800/60 hover:bg-indigo-500/20 border border-gray-700/50 hover:border-indigo-500/30 transition-all"
                    title="Expand"
                  >
                    <Maximize2 className="w-3.5 h-3.5 text-gray-400" />
                  </button>
                  <a
                    href={`${apiUrl}${plot.url}`}
                    download={plot.filename}
                    className="p-1.5 rounded-lg bg-gray-800/60 hover:bg-emerald-500/20 border border-gray-700/50 hover:border-emerald-500/30 transition-all"
                    title="Download"
                  >
                    <Download className="w-3.5 h-3.5 text-gray-400" />
                  </a>
                </div>
              </div>
              <div className="overflow-hidden rounded-xl bg-gray-800/20 border border-gray-700/20 cursor-pointer" onClick={() => setSelectedPlot(plot)}>
                <img
                  src={`${apiUrl}${plot.url}`}
                  alt={plot.filename}
                  className="w-full rounded-xl group-hover:scale-[1.02] transition-transform duration-500"
                  loading="lazy"
                />
              </div>
              <p className="text-[10px] text-gray-600 mt-2 text-right">{(plot.size / 1024).toFixed(0)} KB</p>
            </div>
          ))}
        </div>
      ) : (
        <div className="glass-card p-16 text-center">
          <div className="inline-flex p-5 rounded-2xl bg-gray-800/40 mb-5 border border-gray-700/30">
            <PieChartIcon className="w-12 h-12 text-gray-600" />
          </div>
          <p className="text-gray-400 font-semibold text-lg">No plots generated yet</p>
          <p className="text-sm text-gray-500 mt-2 max-w-md mx-auto">Run the ML pipeline to generate visualizations and they will appear here automatically.</p>
        </div>
      )}

      {/* Lightbox */}
      {selectedPlot && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in" onClick={() => setSelectedPlot(null)}>
          <div className="relative max-w-5xl w-full max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setSelectedPlot(null)}
              className="absolute -top-3 -right-3 z-10 p-2 rounded-full bg-gray-800 border border-gray-700 hover:border-rose-500/50 hover:bg-rose-500/20 transition-all shadow-xl"
            >
              <X className="w-4 h-4 text-gray-300" />
            </button>
            <div className="glass-card p-4 overflow-hidden">
              <p className="text-sm font-semibold text-gray-300 mb-3 capitalize text-center">
                {selectedPlot.filename.replace('.png', '').replace(/_/g, ' ')}
              </p>
              <img
                src={`${apiUrl}${selectedPlot.url}`}
                alt={selectedPlot.filename}
                className="w-full rounded-lg max-h-[80vh] object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
