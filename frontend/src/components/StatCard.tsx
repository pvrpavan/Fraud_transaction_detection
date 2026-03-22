interface StatCardProps {
  icon: React.ElementType;
  label: string;
  value: string;
  subvalue?: string;
  color: string;
  glow?: string;
}

export function StatCard({ icon: Icon, label, value, subvalue, color, glow }: StatCardProps) {
  return (
    <div className={`glass-card p-5 ${glow || ''} animate-slide-up hover:scale-[1.02] transition-all duration-300 group`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">{label}</p>
          <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
          {subvalue && <p className="text-xs text-gray-500 mt-1">{subvalue}</p>}
        </div>
        <div className="p-2.5 rounded-lg bg-gray-800/60 group-hover:bg-gray-800/80 transition-colors">
          <Icon className={`w-5 h-5 ${color}`} />
        </div>
      </div>
    </div>
  );
}
