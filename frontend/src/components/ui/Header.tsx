import { useState, useRef, useEffect } from 'react'
import { Database, Settings, Github, ChevronDown, Check, Plus } from 'lucide-react'
import { cn } from '@/utils/cn'
import { useGraphs, queryKeys } from '@/api/queries'
import { useDatasetStore } from '@/stores/datasetStore'
import { ImportDialog } from '@/components/import/ImportDialog'
import { SettingsDialog } from '@/components/settings/SettingsDialog'
import { Button } from '@/components/ui/button'
import { useQueryClient } from '@tanstack/react-query'

function Header() {
  const [isOpen, setIsOpen] = useState(false)
  const [importDialogOpen, setImportDialogOpen] = useState(false)
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const queryClient = useQueryClient()
  const { data: graphsData, isLoading } = useGraphs()
  const { currentDataset, setCurrentDataset } = useDatasetStore()

  // Get available graphs that have all collections
  const availableGraphs = graphsData?.graphs.filter((g) => g.has_all_collections) || []

  // Auto-select first dataset if none selected
  useEffect(() => {
    if (!currentDataset && availableGraphs.length > 0) {
      setCurrentDataset(availableGraphs[0].name)
    }
  }, [currentDataset, availableGraphs, setCurrentDataset])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const displayName = currentDataset || 'Select dataset...'

  return (
    <header className="h-14 border-b border-slate-200 bg-white px-4 flex items-center justify-between">
      {/* Left section - Logo and title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
          <Database className="w-4 h-4 text-white" />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-slate-800">
            Vector Graph RAG
          </h1>
          <p className="text-xs text-slate-500">Knowledge Graph Explorer</p>
        </div>
      </div>

      {/* Center section - Graph selector */}
      <div className="flex items-center gap-2" ref={dropdownRef}>
        <span className="text-xs text-slate-500">Dataset:</span>
        <div className="relative">
          <button
            onClick={() => setIsOpen(!isOpen)}
            disabled={isLoading}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm min-w-[180px]',
              'border border-slate-200 bg-white',
              'hover:bg-slate-50 transition-colors',
              'flex items-center justify-between gap-2',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            <span className="font-medium text-slate-700 truncate">
              {isLoading ? 'Loading...' : displayName}
            </span>
            <ChevronDown
              className={cn(
                'w-4 h-4 text-slate-400 transition-transform flex-shrink-0',
                isOpen && 'transform rotate-180'
              )}
            />
          </button>

          {/* Dropdown menu */}
          {isOpen && availableGraphs.length > 0 && (
            <div className="absolute top-full left-0 mt-1 w-full min-w-[220px] bg-white border border-slate-200 rounded-md shadow-lg z-50 py-1 max-h-64 overflow-y-auto">
              {availableGraphs.map((graph) => (
                <button
                  key={graph.name}
                  onClick={() => {
                    setCurrentDataset(graph.name)
                    setIsOpen(false)
                  }}
                  className={cn(
                    'w-full px-3 py-2 text-left text-sm',
                    'hover:bg-slate-50 transition-colors',
                    'flex items-center justify-between gap-2',
                    currentDataset === graph.name && 'bg-blue-50'
                  )}
                >
                  <span
                    className={cn(
                      'truncate',
                      currentDataset === graph.name
                        ? 'text-blue-700 font-medium'
                        : 'text-slate-700'
                    )}
                  >
                    {graph.name}
                  </span>
                  {currentDataset === graph.name && (
                    <Check className="w-4 h-4 text-blue-600 flex-shrink-0" />
                  )}
                </button>
              ))}
            </div>
          )}

          {/* Empty state */}
          {isOpen && availableGraphs.length === 0 && !isLoading && (
            <div className="absolute top-full left-0 mt-1 w-full min-w-[220px] bg-white border border-slate-200 rounded-md shadow-lg z-50 py-3 px-3">
              <p className="text-sm text-slate-500 text-center">
                No datasets available
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Right section - Actions */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setImportDialogOpen(true)}
          className="gap-1"
        >
          <Plus className="w-4 h-4" />
          Import
        </Button>
        <a
          href="https://github.com/zilliztech/vector-graph-rag"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          title="GitHub"
        >
          <Github className="w-5 h-5" />
        </a>
        <button
          onClick={() => setSettingsDialogOpen(true)}
          className="p-2 rounded-md text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          title="Settings"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Import Dialog */}
      <ImportDialog
        open={importDialogOpen}
        onClose={() => setImportDialogOpen(false)}
        onImportComplete={() => {
          // Refresh graphs list after import
          queryClient.invalidateQueries({ queryKey: queryKeys.graphs })
        }}
      />

      {/* Settings Dialog */}
      <SettingsDialog
        open={settingsDialogOpen}
        onClose={() => setSettingsDialogOpen(false)}
      />
    </header>
  )
}

export default Header
