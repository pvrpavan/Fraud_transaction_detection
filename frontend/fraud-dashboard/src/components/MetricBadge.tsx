interface MetricBadgeProps {
  label: string;
  value: number;
  good?: boolean;
}

export function MetricBadge({ label, value, good }: MetricBadgeProps) {
  const color = good === undefined ? 'text-gray-300' : good ? 'text-emerald-400' : 'text-rose-400';
  return (
    <div className="flex flex-col items-center p-3 rounded-lg bg-gray-800/40 hover:bg-gray-800/60 transition-colors">
      <span className="text-xs text-gray-400 mb-1">{label}</span>
      <span className={`text-lg font-bold ${color}`}>{(value * 100).toFixed(2)}%</span>
    </div>
  );
}
