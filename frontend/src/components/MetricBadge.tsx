interface MetricBadgeProps {
  label: string;
  value: number;
  good?: boolean;
}

export function MetricBadge({ label, value, good }: MetricBadgeProps) {
  const color = good === undefined ? 'text-gray-300' : good ? 'text-emerald-400' : 'text-amber-400';
  const bgColor = good === undefined ? 'bg-gray-800/40' : good ? 'bg-emerald-500/5' : 'bg-amber-500/5';
  const borderColor = good === undefined ? 'border-gray-700/30' : good ? 'border-emerald-500/20' : 'border-amber-500/20';
  return (
    <div className={`flex flex-col items-center p-3.5 rounded-xl ${bgColor} border ${borderColor} hover:bg-gray-800/60 transition-all duration-200 group`}>
      <span className="text-[10px] text-gray-400 mb-1.5 uppercase tracking-wider font-medium">{label}</span>
      <span className={`text-lg font-bold ${color} group-hover:scale-105 transition-transform`}>{(value * 100).toFixed(2)}%</span>
    </div>
  );
}
