import { cn } from '@/utils/cn'
import { HelpCircle } from 'lucide-react'
import { useState } from 'react'

interface LegendItem {
  label: string
  color: string
  borderColor: string
  description: string
}

const legendItems: LegendItem[] = [
  {
    label: 'Seed',
    color: 'bg-amber-500',
    borderColor: 'border-amber-500',
    description: 'Initial results from vector search',
  },
  {
    label: 'Expanded',
    color: 'bg-blue-500',
    borderColor: 'border-blue-400',
    description: 'Discovered through graph traversal',
  },
  {
    label: 'Selected',
    color: 'bg-emerald-500',
    borderColor: 'border-emerald-500',
    description: 'Chosen by LLM as most relevant',
  },
  {
    label: 'Filtered',
    color: 'bg-slate-400',
    borderColor: 'border-slate-300 border-dashed',
    description: 'Considered but not selected',
  },
]

function GraphLegend() {
  const [showHelp, setShowHelp] = useState(false)

  return (
    <div className="relative flex items-center gap-3 px-3 py-2 bg-white/90 backdrop-blur-sm rounded-lg border border-slate-200 shadow-sm">
      {legendItems.map((item) => (
        <div
          key={item.label}
          className="flex items-center gap-1.5 group relative"
          title={item.description}
        >
          <div
            className={cn(
              'w-4 h-4 rounded border-2 bg-white',
              item.borderColor
            )}
          >
            <div className={cn('w-full h-full rounded-sm', item.color, 'opacity-30')} />
          </div>
          <span className="text-xs font-medium text-slate-600">{item.label}</span>

          {/* Tooltip */}
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-[10px] rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
            {item.description}
          </div>
        </div>
      ))}

      {/* Help button */}
      <button
        onClick={() => setShowHelp(!showHelp)}
        className="ml-1 p-1 text-slate-400 hover:text-slate-600 transition-colors"
        title="How the algorithm works"
      >
        <HelpCircle className="w-4 h-4" />
      </button>

      {/* Help popup */}
      {showHelp && (
        <div className="absolute bottom-full right-0 mb-2 p-3 bg-white rounded-lg border border-slate-200 shadow-lg w-72 text-xs z-50">
          <h4 className="font-semibold text-slate-800 mb-2">How Vector Graph RAG Works</h4>
          <ol className="space-y-1.5 text-slate-600">
            <li><span className="text-amber-600 font-medium">1. Seeds:</span> Vector search finds initial entities & relations</li>
            <li><span className="text-blue-600 font-medium">2. Expand:</span> Graph traversal discovers connected nodes</li>
            <li><span className="text-slate-500 font-medium">3. Filter:</span> Too many? Keep most relevant by similarity</li>
            <li><span className="text-emerald-600 font-medium">4. Rerank:</span> LLM selects best relations for answering</li>
          </ol>
          <p className="mt-2 text-slate-500 text-[10px]">Click timeline steps to see each stage</p>
        </div>
      )}
    </div>
  )
}

export default GraphLegend
