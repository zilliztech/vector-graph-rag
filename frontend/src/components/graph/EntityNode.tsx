import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'
import type { GraphNodeData } from '@/stores/graphStore'

const statusStyles = {
  seed: 'border-amber-500 bg-amber-50 text-amber-700',
  expanded: 'border-blue-500 bg-blue-50 text-blue-700',
  evicted: 'border-slate-400 bg-slate-100 text-slate-500 opacity-60',
  selected: 'border-emerald-500 bg-emerald-50 text-emerald-700 ring-2 ring-emerald-500/30',
  default: 'border-slate-300 bg-white text-slate-700',
}

const EntityNode = memo(({ data, selected }: NodeProps) => {
  const nodeData = data as unknown as GraphNodeData
  const statusStyle = statusStyles[nodeData.status] || statusStyles.default

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      className={cn(
        'px-3 py-2 rounded-lg border-2 shadow-sm cursor-pointer',
        'min-w-[80px] max-w-[160px] text-center',
        'transition-shadow duration-200',
        'hover:shadow-md',
        statusStyle,
        selected && 'ring-2 ring-blue-500/50 shadow-md'
      )}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!w-2 !h-2 !bg-slate-400 !border-0"
      />

      <div className="text-sm font-medium truncate" title={nodeData.label}>
        {nodeData.label}
      </div>

      {nodeData.relationIds.length > 0 && (
        <div className="text-xs opacity-70 mt-0.5">
          {nodeData.relationIds.length} relations
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        className="!w-2 !h-2 !bg-slate-400 !border-0"
      />
    </motion.div>
  )
})

EntityNode.displayName = 'EntityNode'

export default EntityNode
