import { useGraphStats } from '@/api/queries'
import { Box, Network, FileText, Loader2 } from 'lucide-react'

interface GraphStatsPreviewProps {
  graphName: string
}

export function GraphStatsPreview({ graphName }: GraphStatsPreviewProps) {
  const { data: stats, isLoading, error } = useGraphStats(graphName)

  if (isLoading) {
    return (
      <div className="rounded-lg bg-accent/20 border border-slate-200 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin" />
          Loading statistics...
        </div>
      </div>
    )
  }

  if (error || !stats) {
    return (
      <div className="rounded-lg bg-accent/20 border border-slate-200 p-3">
        <div className="text-sm text-slate-500">
          Statistics not available
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-lg bg-accent/20 border border-slate-200 p-3">
      <div className="space-y-2">
        <div className="text-xs font-medium text-slate-600">Current Selection</div>

        <div className="grid grid-cols-3 gap-3 text-xs">
          {/* Entities */}
          <div className="flex items-center gap-1.5">
            <Box className="w-3.5 h-3.5 text-blue-500 flex-shrink-0" />
            <div className="min-w-0">
              <div className="text-slate-500">Entities</div>
              <div className="font-semibold text-slate-700 truncate">
                {stats.entity_count.toLocaleString()}
              </div>
            </div>
          </div>

          {/* Relations */}
          <div className="flex items-center gap-1.5">
            <Network className="w-3.5 h-3.5 text-green-500 flex-shrink-0" />
            <div className="min-w-0">
              <div className="text-slate-500">Relations</div>
              <div className="font-semibold text-slate-700 truncate">
                {stats.relation_count.toLocaleString()}
              </div>
            </div>
          </div>

          {/* Documents */}
          <div className="flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-purple-500 flex-shrink-0" />
            <div className="min-w-0">
              <div className="text-slate-500">Documents</div>
              <div className="font-semibold text-slate-700 truncate">
                {stats.passage_count.toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
