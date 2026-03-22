import { PieChart as PieChartIcon } from 'lucide-react';
import { SectionHeader } from './SectionHeader';
import type { PlotInfo } from '../types';

interface PlotsTabProps {
  plots: PlotInfo[];
  apiUrl: string;
}

export function PlotsTab({ plots, apiUrl }: PlotsTabProps) {
  return (
    <div className="space-y-6 animate-fade-in">
      <SectionHeader icon={PieChartIcon} title="Generated Visualizations" subtitle="ML pipeline output plots" />
      {plots.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {plots.map((plot) => (
            <div key={plot.filename} className="glass-card p-4 group hover:scale-[1.01] transition-all duration-300">
              <p className="text-sm font-medium text-gray-300 mb-3 capitalize">
                {plot.filename.replace('.png', '').replace(/_/g, ' ')}
              </p>
              <div className="overflow-hidden rounded-lg">
                <img
                  src={`${apiUrl}${plot.url}`}
                  alt={plot.filename}
                  className="w-full rounded-lg group-hover:scale-[1.02] transition-transform duration-500"
                  loading="lazy"
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="glass-card p-12 text-center">
          <div className="inline-flex p-4 rounded-full bg-gray-800/40 mb-4">
            <PieChartIcon className="w-10 h-10 text-gray-600" />
          </div>
          <p className="text-gray-400 font-medium">No plots generated yet</p>
          <p className="text-sm text-gray-500 mt-1">Run the ML pipeline to generate visualizations</p>
        </div>
      )}
    </div>
  );
}
