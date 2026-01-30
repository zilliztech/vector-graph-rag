import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Header from '@/components/ui/Header'
import { SearchInput } from '@/components/search'
import { ProcessTimeline } from '@/components/timeline'
import { GraphCanvas, GraphLegend } from '@/components/graph'
import { NodeDetailPanel, AnswerPanel } from '@/components/panels'
import { useSearchStore } from '@/stores/searchStore'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

function AppContent() {
  const { error } = useSearchStore()

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <Header />

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden p-4 gap-4">
        {/* Search bar */}
        <div className="flex-shrink-0">
          <SearchInput />

          {/* Error display */}
          {error && (
            <div className="mt-2 max-w-2xl mx-auto px-4 py-2 bg-red-50 border border-red-200 rounded-md text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        {/* Main content area */}
        <div className="flex-1 flex gap-4 min-h-0">
          {/* Left panel - Timeline */}
          <div className="w-72 flex-shrink-0 bg-white rounded-lg border border-slate-200 overflow-hidden flex flex-col">
            <div className="flex-1 overflow-y-auto">
              <ProcessTimeline />
            </div>
          </div>

          {/* Right area - Graph and Answer */}
          <div className="flex-1 flex flex-col gap-4 min-w-0">
            {/* Graph canvas */}
            <div className="flex-1 relative min-h-0 bg-white rounded-lg border border-slate-200 overflow-hidden">
              <GraphCanvas />

              {/* Legend - positioned at bottom left of canvas */}
              <div className="absolute bottom-4 left-4 z-10">
                <GraphLegend />
              </div>

              {/* Node detail panel - positioned at top right of canvas */}
              <NodeDetailPanel />
            </div>

            {/* Answer panel */}
            <div className="flex-shrink-0">
              <AnswerPanel />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  )
}

export default App
