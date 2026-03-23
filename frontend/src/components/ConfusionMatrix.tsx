interface ConfusionMatrixProps {
  cm: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
}

export function ConfusionMatrix({ cm }: ConfusionMatrixProps) {
  const total = cm.true_negatives + cm.false_positives + cm.false_negatives + cm.true_positives;
  const cells = [
    { label: 'True Negative', shortLabel: 'TN', value: cm.true_negatives, color: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400', hoverColor: 'hover:bg-emerald-500/25' },
    { label: 'False Positive', shortLabel: 'FP', value: cm.false_positives, color: 'bg-amber-500/15 border-amber-500/30 text-amber-400', hoverColor: 'hover:bg-amber-500/25' },
    { label: 'False Negative', shortLabel: 'FN', value: cm.false_negatives, color: 'bg-rose-500/15 border-rose-500/30 text-rose-400', hoverColor: 'hover:bg-rose-500/25' },
    { label: 'True Positive', shortLabel: 'TP', value: cm.true_positives, color: 'bg-indigo-500/15 border-indigo-500/30 text-indigo-400', hoverColor: 'hover:bg-indigo-500/25' },
  ];

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center text-[10px] text-gray-500 uppercase tracking-wider px-1">
        <span className="bg-gray-800/40 px-2 py-1 rounded-md">Actual ↓</span>
        <span className="bg-gray-800/40 px-2 py-1 rounded-md">Predicted →</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {cells.map((cell) => (
          <div key={cell.label} className={`${cell.color} ${cell.hoverColor} border rounded-xl p-4 text-center transition-all duration-200 cursor-default group`}>
            <div className="flex items-center justify-center gap-1.5 mb-1.5">
              <span className="text-[10px] font-bold opacity-60">{cell.shortLabel}</span>
              <p className="text-xs text-gray-400">{cell.label}</p>
            </div>
            <p className="text-2xl font-bold group-hover:scale-105 transition-transform">{cell.value.toLocaleString()}</p>
            <p className="text-xs text-gray-500 mt-1.5 font-mono">{(cell.value / total * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}
