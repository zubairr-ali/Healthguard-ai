export default function ShapBars({ contributions }) {
  if (!contributions?.length) return null;
  const max = Math.max(...contributions.map((c) => Math.abs(c.value)), 0.001);

  return (
    <div className="space-y-3">
      {contributions.map((c) => {
        const pct = (Math.abs(c.value) / max) * 100;
        const positive = c.value >= 0;
        return (
          <div key={c.feature} className="grid grid-cols-[minmax(0,1fr)_2.6fr_auto] items-center gap-3">
            <span className="text-sm text-ink-300 light:text-ink-600 truncate" title={c.feature}>
              {c.feature}
            </span>
            <div className="relative h-6 bg-ink-800/60 light:bg-ink-100 rounded-md overflow-hidden">
              <div className="absolute inset-y-0 left-1/2 w-px bg-ink-600 light:bg-ink-300" />
              <div
                className={`absolute inset-y-0 rounded-md ${positive ? 'bg-signal-500' : 'bg-vital-500'}`}
                style={{
                  width: `${pct / 2}%`,
                  left: positive ? '50%' : `${50 - pct / 2}%`,
                }}
              />
            </div>
            <span
              className={`data-readout text-xs w-14 text-right ${
                positive ? 'text-signal-400' : 'text-vital-400'
              }`}
            >
              {positive ? '+' : ''}
              {c.value.toFixed(3)}
            </span>
          </div>
        );
      })}
      <div className="flex items-center gap-4 pt-1 text-xs text-ink-400 light:text-ink-500">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-signal-500 inline-block" /> increases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-vital-500 inline-block" /> decreases risk
        </span>
      </div>
    </div>
  );
}
