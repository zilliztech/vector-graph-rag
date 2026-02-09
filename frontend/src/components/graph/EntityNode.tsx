import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'
import type { GraphNodeData } from '@/stores/graphStore'

// Enhanced status styles with more visual distinction
const statusStyles = {
  // Seed: Orange/amber - initial vector search results
  seed: {
    container: 'border-amber-500 border-[3px] bg-amber-50 shadow-amber-200/50',
    text: 'text-amber-800 font-semibold',
    badge: 'bg-amber-500 text-white',
    glow: 'ring-2 ring-amber-300/50',
  },
  // Expanded: Blue - discovered through graph traversal
  expanded: {
    container: 'border-blue-400 border-2 bg-blue-50 shadow-blue-200/50',
    text: 'text-blue-700',
    badge: 'bg-blue-400 text-white',
    glow: '',
  },
  // Selected: Green - chosen by LLM rerank (entities connected to selected relations)
  selected: {
    container: 'border-emerald-500 border-[3px] bg-emerald-50 shadow-emerald-200/50',
    text: 'text-emerald-800 font-semibold',
    badge: 'bg-emerald-500 text-white',
    glow: 'ring-2 ring-emerald-300/50',
  },
  // Filtered: Gray - was considered but not selected
  filtered: {
    container: 'border-slate-300 border-dashed border-2 bg-slate-50 opacity-50',
    text: 'text-slate-500',
    badge: 'bg-slate-300 text-slate-600',
    glow: '',
  },
  // Undiscovered: Very faded - not yet discovered at this step
  undiscovered: {
    container: 'border-slate-200 border bg-slate-50/50 opacity-30',
    text: 'text-slate-400',
    badge: 'bg-slate-200 text-slate-400',
    glow: '',
  },
}

const EntityNode = memo(({ data, selected }: NodeProps) => {
  const nodeData = data as unknown as GraphNodeData
  const styles = statusStyles[nodeData.status] || statusStyles.undiscovered

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{
        scale: 1,
        opacity: nodeData.status === 'undiscovered' ? 0.3 :
                 nodeData.status === 'filtered' ? 0.5 : 1
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      className={cn(
        'px-3 py-2 rounded-lg shadow-md cursor-pointer',
        'min-w-[90px] max-w-[180px] text-center',
        'transition-all duration-300',
        'hover:shadow-lg hover:scale-105',
        styles.container,
        styles.glow,
        selected && 'ring-2 ring-blue-500 shadow-lg scale-105'
      )}
    >
      <Handle
        type="target"
        position={Position.Left}
        className={cn(
          '!w-2 !h-2 !border-0',
          nodeData.status === 'seed' ? '!bg-amber-500' :
          nodeData.status === 'selected' ? '!bg-emerald-500' :
          nodeData.status === 'expanded' ? '!bg-blue-400' :
          '!bg-slate-300'
        )}
      />

      <div className={cn('text-sm truncate', styles.text)} title={nodeData.label}>
        {nodeData.label}
      </div>

      {nodeData.relationIds.length > 0 && (
        <div className={cn(
          'text-xs mt-1 px-1.5 py-0.5 rounded-full inline-block',
          styles.badge
        )}>
          {nodeData.relationIds.length} relations
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        className={cn(
          '!w-2 !h-2 !border-0',
          nodeData.status === 'seed' ? '!bg-amber-500' :
          nodeData.status === 'selected' ? '!bg-emerald-500' :
          nodeData.status === 'expanded' ? '!bg-blue-400' :
          '!bg-slate-300'
        )}
      />
    </motion.div>
  )
})

EntityNode.displayName = 'EntityNode'

export default EntityNode
