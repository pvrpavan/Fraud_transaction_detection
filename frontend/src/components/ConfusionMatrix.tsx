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
    { label: 'True Negative', value: cm.true_negatives, color: 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400', hoverColor: 'hover:bg-emerald-500/30' },
    { label: 'False Positive', value: cm.false_positives, color: 'bg-amber-500/20 border-amber-500/30 text-amber-400', hoverColor: 'hover:bg-amber-500/30' },
    { label: 'False Negative', value: cm.false_negatives, color: 'bg-rose-500/20 border-rose-500/30 text-rose-400', hoverColor: 'hover:bg-rose-500/30' },
    { label: 'True Positive', value: cm.true_positives, color: 'bg-indigo-500/20 border-indigo-500/30 text-indigo-400', hoverColor: 'hover:bg-indigo-500/30' },
  ];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-1 text-center text-xs text-gray-500 mb-1">
        <div className="col-span-2 flex justify-center">
          <span className="px-3 py-1 rounded-full bg-gray-800/40 text-gray-400 text-[10px] uppercase tracking-wider font-medium">Predicted →</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {cells.map((cell) => (
          <div key={cell.label} className={`${cell.color} ${cell.hoverColor} border rounded-lg p-4 text-center transition-all duration-200 cursor-default`}>
            <p className="text-xs text-gray-400 mb-1">{cell.label}</p>
            <p className="text-xl font-bold">{cell.value.toLocaleString()}</p>
            <p className="text-xs text-gray-500 mt-1">{(cell.value / total * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}
