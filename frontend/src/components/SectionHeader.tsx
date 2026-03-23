interface SectionHeaderProps {
  icon: React.ElementType;
  title: string;
  subtitle?: string;
  accent?: 'indigo' | 'emerald' | 'rose' | 'amber' | 'purple';
}

const accentStyles = {
  indigo: 'bg-indigo-500/10 border-indigo-500/20 text-indigo-400',
  emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
  rose: 'bg-rose-500/10 border-rose-500/20 text-rose-400',
  amber: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
  purple: 'bg-purple-500/10 border-purple-500/20 text-purple-400',
};

export function SectionHeader({ icon: Icon, title, subtitle, accent = 'indigo' }: SectionHeaderProps) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <div className={`p-2 rounded-xl border ${accentStyles[accent]}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <h2 className="text-lg font-bold text-white">{title}</h2>
        {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}
