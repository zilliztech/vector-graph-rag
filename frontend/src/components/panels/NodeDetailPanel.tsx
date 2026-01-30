import { useMemo } from 'react'
import { X, Link2, FileText } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useGraphStore } from '@/stores/graphStore'

function NodeDetailPanel() {
  const { selectedNodeId, subgraph, setSelectedNode } = useGraphStore()

  const selectedEntity = useMemo(() => {
    if (!selectedNodeId || !subgraph) return null
    return subgraph.entities.find((e) => e.id === selectedNodeId)
  }, [selectedNodeId, subgraph])

  const connectedRelations = useMemo(() => {
    if (!selectedEntity || !subgraph) return []
    return subgraph.relations.filter((r) =>
      r.entity_ids.includes(selectedEntity.id)
    )
  }, [selectedEntity, subgraph])

  const connectedPassages = useMemo(() => {
    if (!selectedEntity || !subgraph) return []
    return subgraph.passages.filter((p) =>
      selectedEntity.passage_ids.includes(p.id)
    )
  }, [selectedEntity, subgraph])

  return (
    <AnimatePresence>
      {selectedEntity && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 20 }}
          className="absolute right-4 top-4 w-80 bg-white rounded-lg border border-slate-200 shadow-lg overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-slate-50">
            <h3 className="font-medium text-slate-700 truncate flex-1">
              {selectedEntity.name}
            </h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 rounded hover:bg-slate-200 text-slate-500 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Content */}
          <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
            {/* Connected Relations */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Link2 className="w-4 h-4 text-slate-400" />
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Relations ({connectedRelations.length})
                </span>
              </div>
              <div className="space-y-2">
                {connectedRelations.length > 0 ? (
                  connectedRelations.slice(0, 5).map((relation) => (
                    <div
                      key={relation.id}
                      className="p-2 bg-slate-50 rounded text-xs text-slate-600"
                    >
                      <span className="font-medium">{relation.subject}</span>
                      <span className="mx-1 text-violet-600">{relation.predicate}</span>
                      <span className="font-medium">{relation.object}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-400">No relations</p>
                )}
                {connectedRelations.length > 5 && (
                  <p className="text-xs text-slate-400">
                    +{connectedRelations.length - 5} more
                  </p>
                )}
              </div>
            </div>

            {/* Connected Passages */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-slate-400" />
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Sources ({connectedPassages.length})
                </span>
              </div>
              <div className="space-y-2">
                {connectedPassages.length > 0 ? (
                  connectedPassages.slice(0, 3).map((passage) => (
                    <div
                      key={passage.id}
                      className="p-2 bg-slate-50 rounded text-xs text-slate-600 line-clamp-3"
                    >
                      {passage.text}
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-400">No source passages</p>
                )}
                {connectedPassages.length > 3 && (
                  <p className="text-xs text-slate-400">
                    +{connectedPassages.length - 3} more
                  </p>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default NodeDetailPanel
