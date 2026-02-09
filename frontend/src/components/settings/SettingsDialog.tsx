import { useState } from 'react'
import { X, Database, Cpu, Layers, Server, Key, Trash2, AlertTriangle } from 'lucide-react'
import { useSettings, useGraphs, useDeleteGraph } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { cn } from '@/utils/cn'

interface SettingsDialogProps {
  open: boolean
  onClose: () => void
}

export function SettingsDialog({ open, onClose }: SettingsDialogProps) {
  const { data: settings, isLoading: settingsLoading } = useSettings()
  const { data: graphsData, isLoading: graphsLoading } = useGraphs()
  const deleteGraphMutation = useDeleteGraph()
  const [deletingGraph, setDeletingGraph] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  if (!open) return null

  const handleDeleteGraph = async (graphName: string) => {
    if (confirmDelete !== graphName) {
      setConfirmDelete(graphName)
      return
    }

    setDeletingGraph(graphName)
    try {
      await deleteGraphMutation.mutateAsync(graphName)
      setConfirmDelete(null)
    } catch (error) {
      console.error('Failed to delete graph:', error)
    } finally {
      setDeletingGraph(null)
    }
  }

  const cancelDelete = () => {
    setConfirmDelete(null)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="relative w-full max-w-3xl max-h-[90vh] bg-white rounded-lg shadow-xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
              <Server className="w-4 h-4 text-white" />
            </div>
            <h2 className="text-lg font-semibold text-slate-800">System Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
          {/* System Configuration */}
          <section>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Configuration</h3>
            {settingsLoading ? (
              <div className="text-sm text-slate-500">Loading settings...</div>
            ) : settings ? (
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                  <Cpu className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-slate-600">LLM Model</div>
                    <div className="text-sm text-slate-800 font-mono">{settings.llm_model}</div>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                  <Layers className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-slate-600">Embedding Model</div>
                    <div className="text-sm text-slate-800 font-mono truncate">
                      {settings.embedding_model}
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      Dimension: {settings.embedding_dimension}
                    </div>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                  <Database className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-slate-600">Milvus Connection</div>
                    <div className="text-sm text-slate-800 font-mono truncate">
                      {settings.milvus_uri}
                    </div>
                    {settings.milvus_db && (
                      <div className="text-xs text-slate-500 mt-0.5">
                        Database: {settings.milvus_db}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                  <Key className="w-4 h-4 text-orange-600 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-slate-600">OpenAI API Key</div>
                    <div className="text-sm text-slate-800">
                      {settings.openai_api_key_set ? (
                        <span className="text-green-600 font-medium">✓ Configured</span>
                      ) : (
                        <span className="text-red-600 font-medium">✗ Not set</span>
                      )}
                    </div>
                    {settings.openai_base_url && (
                      <div className="text-xs text-slate-500 mt-0.5 font-mono truncate">
                        Base URL: {settings.openai_base_url}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-sm text-red-600">Failed to load settings</div>
            )}
          </section>

          {/* Knowledge Bases Management */}
          <section>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Knowledge Bases</h3>
            {graphsLoading ? (
              <div className="text-sm text-slate-500">Loading knowledge bases...</div>
            ) : graphsData?.graphs && graphsData.graphs.length > 0 ? (
              <div className="space-y-2">
                {graphsData.graphs
                  .filter((g) => g.has_all_collections)
                  .map((graph) => (
                    <div
                      key={graph.name}
                      className={cn(
                        'flex items-center justify-between p-3 rounded-lg border transition-all',
                        confirmDelete === graph.name
                          ? 'border-red-300 bg-red-50'
                          : 'border-slate-200 bg-white hover:bg-slate-50'
                      )}
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <Database className="w-4 h-4 text-blue-600 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-slate-800 truncate">
                            {graph.name}
                          </div>
                          <div className="text-xs text-slate-500">
                            {graph.entity_collection} · {graph.relation_collection} ·{' '}
                            {graph.passage_collection}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {confirmDelete === graph.name ? (
                          <>
                            <div className="flex items-center gap-2 mr-2">
                              <AlertTriangle className="w-4 h-4 text-red-600" />
                              <span className="text-xs font-medium text-red-600">
                                Confirm delete?
                              </span>
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={cancelDelete}
                              disabled={deletingGraph === graph.name}
                              className="h-7 px-2 text-xs"
                            >
                              Cancel
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => handleDeleteGraph(graph.name)}
                              disabled={deletingGraph === graph.name}
                              className="h-7 px-2 text-xs bg-red-600 hover:bg-red-700 text-white"
                            >
                              {deletingGraph === graph.name ? 'Deleting...' : 'Yes, Delete'}
                            </Button>
                          </>
                        ) : (
                          <button
                            onClick={() => handleDeleteGraph(graph.name)}
                            disabled={deletingGraph !== null}
                            className={cn(
                              'p-1.5 rounded-md transition-colors',
                              deletingGraph !== null
                                ? 'text-slate-300 cursor-not-allowed'
                                : 'text-slate-400 hover:text-red-600 hover:bg-red-50'
                            )}
                            title="Delete knowledge base"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <div className="text-sm text-slate-500">No knowledge bases found</div>
            )}
          </section>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-200 bg-slate-50">
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-500">
              Settings are read-only. Modify via environment variables or config files.
            </p>
            <Button onClick={onClose} size="sm">
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
