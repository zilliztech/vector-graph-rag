import { memo, useState, useCallback, useRef } from 'react'
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'
import type { GraphEdgeData } from '@/stores/graphStore'

// Store for persisting label offsets across renders
const labelOffsets = new Map<string, { x: number; y: number }>()

// Enhanced status styles for edges
const edgeStyles = {
  // Seed: Orange/amber - initial vector search results
  seed: {
    color: '#f59e0b', // amber-500
    width: 3,
    opacity: 1,
    dashArray: undefined,
    glow: true,
    labelBg: 'bg-amber-50 border-amber-300',
  },
  // Expanded: Blue - discovered through graph traversal
  expanded: {
    color: '#3b82f6', // blue-500
    width: 2,
    opacity: 1,
    dashArray: undefined,
    glow: false,
    labelBg: 'bg-blue-50 border-blue-300',
  },
  // Selected: Green - chosen by LLM rerank
  selected: {
    color: '#10b981', // emerald-500
    width: 4,
    opacity: 1,
    dashArray: undefined,
    glow: true,
    labelBg: 'bg-emerald-50 border-emerald-400',
  },
  // Filtered: Gray dashed - was considered but not selected
  filtered: {
    color: '#94a3b8', // slate-400
    width: 1.5,
    opacity: 0.5,
    dashArray: '6,4',
    glow: false,
    labelBg: 'bg-slate-50 border-slate-200',
  },
  // Undiscovered: Very faded - not yet discovered at this step
  undiscovered: {
    color: '#cbd5e1', // slate-300
    width: 1,
    opacity: 0.2,
    dashArray: '2,4',
    glow: false,
    labelBg: 'bg-slate-50 border-slate-100',
  },
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
    const [isDragging, setIsDragging] = useState(false)
    const edgeData = data as unknown as GraphEdgeData

    // Get or initialize label offset for this edge
    const [dragOffset, setDragOffset] = useState(() => {
      return labelOffsets.get(id) || { x: 0, y: 0 }
    })

    // Refs for drag handling
    const dragStartRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null)

    // Calculate offset for multiple edges between same node pair
    const edgeIndex = edgeData?.edgeIndex || 0
    const totalEdges = edgeData?.totalEdges || 1

    // For single edge, use small curvature
    // For multiple edges, use different curvature values to separate them
    // Note: getBezierPath only accepts positive curvature values
    let curvature = 0.2 // default for single edge

    if (totalEdges > 1) {
      // Spread curvature values from 0.1 to 0.5 based on edge index
      // This creates visible separation between parallel edges
      const minCurvature = 0.1
      const maxCurvature = 0.5
      const step = (maxCurvature - minCurvature) / Math.max(1, totalEdges - 1)
      curvature = minCurvature + edgeIndex * step
    }

    // Get default path and label position from getBezierPath
    const [defaultEdgePath, defaultLabelX, defaultLabelY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      curvature,
    })

    // Drag handlers for label repositioning
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
      e.stopPropagation()
      e.preventDefault()
      setIsDragging(true)
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        offsetX: dragOffset.x,
        offsetY: dragOffset.y,
      }

      const handleMouseMove = (moveEvent: MouseEvent) => {
        if (!dragStartRef.current) return
        const dx = moveEvent.clientX - dragStartRef.current.x
        const dy = moveEvent.clientY - dragStartRef.current.y
        const newOffset = {
          x: dragStartRef.current.offsetX + dx,
          y: dragStartRef.current.offsetY + dy,
        }
        setDragOffset(newOffset)
        labelOffsets.set(id, newOffset)
      }

      const handleMouseUp = () => {
        setIsDragging(false)
        dragStartRef.current = null
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }

      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }, [id, dragOffset])

    // Double-click to reset position
    const handleDoubleClick = useCallback((e: React.MouseEvent) => {
      e.stopPropagation()
      e.preventDefault()
      setDragOffset({ x: 0, y: 0 })
      labelOffsets.delete(id)
    }, [id])

    // Calculate final label position with drag offset
    const labelX = defaultLabelX + dragOffset.x
    const labelY = defaultLabelY + dragOffset.y

    // Create custom edge path that passes through the label position
    // Use quadratic bezier curve: M start Q control end
    // The control point is calculated to make the curve pass through the label
    // For a quadratic bezier, if we want it to pass through point P at t=0.5:
    // P = (1-t)²·start + 2(1-t)t·control + t²·end
    // At t=0.5: P = 0.25·start + 0.5·control + 0.25·end
    // So: control = 2·P - 0.5·start - 0.5·end
    const hasOffset = dragOffset.x !== 0 || dragOffset.y !== 0

    let edgePath: string
    if (hasOffset) {
      // Calculate control point so curve passes through label position
      const controlX = 2 * labelX - 0.5 * sourceX - 0.5 * targetX
      const controlY = 2 * labelY - 0.5 * sourceY - 0.5 * targetY
      edgePath = `M ${sourceX} ${sourceY} Q ${controlX} ${controlY} ${targetX} ${targetY}`
    } else {
      edgePath = defaultEdgePath
    }

    // No additional offset needed since labelX/labelY already include dragOffset

    const status = edgeData?.status || 'undiscovered'
    const styles = edgeStyles[status] || edgeStyles.undiscovered

    // Get connected node statuses
    const sourceNodeStatus = edgeData?.sourceNodeStatus || 'undiscovered'
    const targetNodeStatus = edgeData?.targetNodeStatus || 'undiscovered'

    // Only show label if BOTH connected nodes are visible (not undiscovered)
    // This prevents labels from appearing in weird positions when one node is invisible
    // We show labels even for 'undiscovered' edges as long as both nodes are visible
    // so users can see all relationships between visible entities
    const bothNodesVisible = sourceNodeStatus !== 'undiscovered' && targetNodeStatus !== 'undiscovered'
    const showLabel = bothNodesVisible

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

        {/* Animated glow for highlighted edges */}
        {styles.glow && (
          <motion.path
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.2, 0.5, 0.2] }}
            transition={{ duration: 2, repeat: Infinity }}
            d={edgePath}
            fill="none"
            stroke={styles.color}
            strokeWidth={styles.width + 6}
            style={{ filter: 'blur(4px)' }}
          />
        )}

        {/* Visible edge */}
        <BaseEdge
          id={id}
          path={edgePath}
          style={{
            stroke: styles.color,
            strokeWidth: styles.width,
            opacity: styles.opacity,
            strokeDasharray: styles.dashArray,
            transition: 'all 0.3s ease',
          }}
        />

        {/* Arrow marker for selected edges */}
        {status === 'selected' && (
          <defs>
            <marker
              id={`arrow-${id}`}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={styles.color} />
            </marker>
          </defs>
        )}

        {/* Edge label */}
        {showLabel && (
          <EdgeLabelRenderer>
            {/* Outer div for positioning - must be separate from motion.div */}
            {/* because framer-motion's scale animation overwrites the transform property */}
            <div
              style={{
                position: 'absolute',
                transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                pointerEvents: 'all',
                cursor: isDragging ? 'grabbing' : 'grab',
                zIndex: isDragging ? 1000 : undefined,
              }}
              onMouseDown={handleMouseDown}
              onDoubleClick={handleDoubleClick}
              title="Drag to reposition, double-click to reset"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{
                  opacity: status === 'filtered' ? 0.5 :
                           isHovered || selected || isDragging ? 1 : 0.85,
                  scale: isHovered && !isDragging ? 1.08 : 1,
                }}
                transition={{ duration: 0.2 }}
                className={cn(
                  'px-2 py-0.5 rounded-md text-xs font-medium',
                  'border shadow-sm',
                  'transition-all duration-200',
                  styles.labelBg,
                  status === 'selected' && 'ring-1 ring-emerald-400/50',
                  status === 'seed' && 'ring-1 ring-amber-400/50',
                  isHovered && !isDragging && 'shadow-md',
                  isDragging && 'shadow-lg ring-2 ring-blue-400'
                )}
                onMouseEnter={() => !isDragging && setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
              >
              <span
                style={{ color: styles.color }}
                className={cn(
                  status === 'selected' && 'font-semibold'
                )}
              >
                {edgeData?.predicate || 'relates'}
              </span>

              {/* Tooltip on hover */}
              {isHovered && edgeData && (
                <div
                  className={cn(
                    'absolute left-1/2 -translate-x-1/2 top-full mt-2',
                    'px-3 py-2 rounded-lg bg-slate-800 text-white text-xs',
                    'whitespace-nowrap shadow-lg z-50',
                    'min-w-[120px]'
                  )}
                >
                  <div className="font-medium text-slate-100">{edgeData.subject}</div>
                  <div className="text-amber-400 my-0.5 text-center">↓ {edgeData.predicate}</div>
                  <div className="font-medium text-slate-100">{edgeData.object}</div>
                  {status === 'filtered' && (
                    <div className="mt-1 pt-1 border-t border-slate-600 text-slate-400 text-[10px]">
                      Filtered out
                    </div>
                  )}
                  {status === 'selected' && (
                    <div className="mt-1 pt-1 border-t border-slate-600 text-emerald-400 text-[10px]">
                      ✓ Selected by LLM
                    </div>
                  )}
                </div>
              )}
              </motion.div>
            </div>
          </EdgeLabelRenderer>
        )}
      </>
    )
  }
)

RelationEdge.displayName = 'RelationEdge'

export default RelationEdge
