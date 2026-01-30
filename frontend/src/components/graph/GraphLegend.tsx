import { cn } from '@/utils/cn'

interface LegendItem {
  label: string
  color: string
  bgColor: string
}

const legendItems: LegendItem[] = [
  { label: 'Seed', color: 'bg-amber-500', bgColor: 'bg-amber-50' },
  { label: 'Expanded', color: 'bg-blue-500', bgColor: 'bg-blue-50' },
  { label: 'Selected', color: 'bg-emerald-500', bgColor: 'bg-emerald-50' },
  { label: 'Filtered', color: 'bg-slate-400', bgColor: 'bg-slate-100' },
]

function GraphLegend() {
  return (
    <div className="flex items-center gap-4 px-3 py-2 bg-white/80 backdrop-blur-sm rounded-lg border border-slate-200 shadow-sm">
      {legendItems.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5">
          <div
            className={cn(
              'w-3 h-3 rounded-full border-2',
              item.bgColor
            )}
            style={{ borderColor: item.color.replace('bg-', 'var(--color-') }}
          >
            <div className={cn('w-full h-full rounded-full', item.color, 'opacity-50')} />
          </div>
          <span className="text-xs text-slate-600">{item.label}</span>
        </div>
      ))}
    </div>
  )
}

export default GraphLegend
