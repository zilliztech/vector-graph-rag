import { memo, useState } from 'react'
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'
import type { GraphEdgeData } from '@/stores/graphStore'

const statusColors = {
  seed: '#f59e0b', // amber-500
  expanded: '#3b82f6', // blue-500
  evicted: '#94a3b8', // slate-400
  selected: '#10b981', // emerald-500
  default: '#64748b', // slate-500
}

const RelationEdge = memo(
  ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    data,
    selected,
  }: EdgeProps) => {
    const [isHovered, setIsHovered] = useState(false)
    const edgeData = data as unknown as GraphEdgeData

    const [edgePath, labelX, labelY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    })

    const strokeColor = statusColors[edgeData?.status] || statusColors.default
    const isHighlighted = edgeData?.status === 'selected' || edgeData?.status === 'seed'

    return (
      <>
        {/* Invisible wider path for easier hover detection */}
        <path
          d={edgePath}
          fill="none"
          stroke="transparent"
          strokeWidth={20}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          style={{ cursor: 'pointer' }}
        />

        {/* Visible edge */}
        <BaseEdge
          id={id}
          path={edgePath}
          style={{
            stroke: strokeColor,
            strokeWidth: isHighlighted ? 2.5 : 1.5,
            opacity: edgeData?.status === 'evicted' ? 0.4 : 1,
          }}
        />

        {/* Animated glow for selected edges */}
        {isHighlighted && (
          <motion.path
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 2, repeat: Infinity }}
            d={edgePath}
            fill="none"
            stroke={strokeColor}
            strokeWidth={6}
            style={{ filter: 'blur(3px)' }}
          />
        )}

        {/* Edge label */}
        <EdgeLabelRenderer>
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{
              opacity: isHovered || selected ? 1 : 0.7,
              scale: isHovered ? 1.05 : 1,
            }}
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
            className={cn(
              'px-2 py-0.5 rounded text-xs font-medium',
              'bg-white border shadow-sm',
              'cursor-pointer transition-colors',
              isHovered && 'bg-slate-50'
            )}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
          >
            <span style={{ color: strokeColor }}>{edgeData?.predicate || 'relates'}</span>

            {/* Tooltip on hover */}
            {isHovered && edgeData && (
              <div
                className={cn(
                  'absolute left-1/2 -translate-x-1/2 top-full mt-2',
                  'px-3 py-2 rounded-lg bg-slate-800 text-white text-xs',
                  'whitespace-nowrap shadow-lg z-50'
                )}
              >
                <div className="font-medium">{edgeData.subject}</div>
                <div className="text-slate-300 my-0.5">{edgeData.predicate}</div>
                <div className="font-medium">{edgeData.object}</div>
              </div>
            )}
          </motion.div>
        </EdgeLabelRenderer>
      </>
    )
  }
)

RelationEdge.displayName = 'RelationEdge'

export default RelationEdge
