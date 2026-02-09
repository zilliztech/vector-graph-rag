import { useState } from 'react'
import { Database, Check, ChevronDown, Plus } from 'lucide-react'
import { cn } from '@/utils/cn'
import { useGraphs } from '@/api/queries'

interface GraphSelectorProps {
  value: string
  onChange: (graphName: string) => void
  onCreateNew?: () => void
}

export function GraphSelector({ value, onChange, onCreateNew }: GraphSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const { data: graphsData, isLoading } = useGraphs()

  const availableGraphs = graphsData?.graphs.filter((g) => g.has_all_collections) || []
  const selectedGraph = availableGraphs.find((g) => g.name === value)

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-slate-700 flex items-center gap-2">
        <Database className="w-4 h-4" />
        Knowledge Base
      </label>

      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          disabled={isLoading}
          className={cn(
            'w-full px-3 py-2 rounded-md text-sm',
            'border border-slate-200 bg-white',
            'hover:bg-slate-50 transition-colors',
            'flex items-center justify-between gap-2',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
          )}
        >
          <span className="font-medium text-slate-700 truncate">
            {isLoading ? 'Loading...' : selectedGraph?.name || value || 'Select knowledge base'}
          </span>
          <ChevronDown
            className={cn(
              'w-4 h-4 text-slate-400 transition-transform flex-shrink-0',
              isOpen && 'transform rotate-180'
            )}
          />
        </button>

        {/* Dropdown menu */}
        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-10"
              onClick={() => setIsOpen(false)}
            />

            {/* Menu */}
            <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-slate-200 rounded-md shadow-lg z-20 max-h-64 overflow-y-auto">
              {availableGraphs.length > 0 ? (
                <div className="py-1">
                  {availableGraphs.map((graph) => (
                    <button
                      key={graph.name}
                      type="button"
                      onClick={() => {
                        onChange(graph.name)
                        setIsOpen(false)
                      }}
                      className={cn(
                        'w-full px-3 py-2 text-left text-sm',
                        'hover:bg-slate-50 transition-colors',
                        'flex items-center justify-between gap-2',
                        value === graph.name && 'bg-blue-50'
                      )}
                    >
                      <div className="flex-1 min-w-0">
                        <div
                          className={cn(
                            'font-medium truncate',
                            value === graph.name ? 'text-blue-700' : 'text-slate-700'
                          )}
                        >
                          {graph.name}
                        </div>
                        <div className="text-xs text-slate-500">
                          {/* We'll add stats later */}
                          Knowledge base
                        </div>
                      </div>
                      {value === graph.name && (
                        <Check className="w-4 h-4 text-blue-600 flex-shrink-0" />
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="px-3 py-4 text-sm text-slate-500 text-center">
                  No knowledge bases available
                </div>
              )}

              {/* Create new option */}
              {onCreateNew && (
                <>
                  <div className="border-t border-slate-200" />
                  <button
                    type="button"
                    onClick={() => {
                      onCreateNew()
                      setIsOpen(false)
                    }}
                    className="w-full px-3 py-2 text-left text-sm text-blue-600 hover:bg-blue-50 transition-colors flex items-center gap-2"
                  >
                    <Plus className="w-4 h-4" />
                    Create New Knowledge Base
                  </button>
                </>
              )}
            </div>
          </>
        )}
      </div>

      {/* Current selection info */}
      {selectedGraph && (
        <div className="text-xs text-slate-500">
          <span className="inline-flex items-center gap-1">
            💡 Importing to: <span className="font-medium text-slate-700">{selectedGraph.name}</span>
          </span>
        </div>
      )}
    </div>
  )
}
