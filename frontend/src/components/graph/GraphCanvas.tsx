import { useCallback, useEffect } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore, type GraphNodeData } from '@/stores/graphStore'
import { layoutGraph, forceDirectedLayout } from '@/utils/graphLayout'
import EntityNode from './EntityNode'
import RelationEdge from './RelationEdge'

const nodeTypes = {
  entity: EntityNode,
}

const edgeTypes = {
  relation: RelationEdge,
}

const minimapStyle = {
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  borderRadius: '6px',
}

function GraphCanvas() {
  const {
    nodes: storeNodes,
    edges: storeEdges,
    setSelectedNode,
    setHoveredNode,
  } = useGraphStore()

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])

  // Apply layout when store nodes/edges change
  useEffect(() => {
    if (storeNodes.length === 0) {
      setNodes([])
      setEdges([])
      return
    }

    // Use force-directed layout for larger graphs (better space utilization)
    // Use dagre layout for smaller graphs (cleaner hierarchy)
    const useForceLayout = storeNodes.length > 30 || storeEdges.length > 50

    // Make the layout area larger (4x screen size) so the graph isn't cramped
    const layoutedNodes = useForceLayout
      ? forceDirectedLayout(storeNodes, storeEdges, {
          width: 4000,
          height: 2500,
          iterations: 200,
          repulsion: 8000,
          attraction: 0.005,
        })
      : layoutGraph(storeNodes, storeEdges, {
          direction: 'LR',
          nodeSpacing: 50,
          rankSpacing: 120,
        })

    setNodes(layoutedNodes as Node[])
    setEdges(storeEdges as Edge[])
  }, [storeNodes, storeEdges, setNodes, setEdges])

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node.id)
    },
    [setSelectedNode]
  )

  const onNodeMouseEnter = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setHoveredNode(node.id)
    },
    [setHoveredNode]
  )

  const onNodeMouseLeave = useCallback(() => {
    setHoveredNode(null)
  }, [setHoveredNode])

  const onPaneClick = useCallback(() => {
    setSelectedNode(null)
  }, [setSelectedNode])

  // Minimap node color based on status
  const nodeColor = useCallback((node: Node) => {
    const data = node.data as GraphNodeData | undefined
    switch (data?.status) {
      case 'seed':
        return '#f59e0b' // amber
      case 'selected':
        return '#10b981' // emerald
      case 'expanded':
        return '#3b82f6' // blue
      case 'filtered':
        return '#94a3b8' // slate
      case 'undiscovered':
        return '#e2e8f0' // light slate
      default:
        return '#3b82f6'
    }
  }, [])

  // Empty state
  if (nodes.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-50 rounded-lg border border-slate-200">
        <div className="text-center text-slate-500">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-slate-300"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          <p className="text-sm font-medium">No graph to display</p>
          <p className="text-xs mt-1">Enter a query to visualize the knowledge graph</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onNodeMouseEnter={onNodeMouseEnter}
        onNodeMouseLeave={onNodeMouseLeave}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.1, maxZoom: 0.6 }}
        minZoom={0.05}
        maxZoom={2}
        defaultEdgeOptions={{
          type: 'relation',
        }}
      >
        <Background color="#e2e8f0" gap={20} size={1} />
        <Controls
          className="bg-white border border-slate-200 rounded-md shadow-sm"
          showInteractive={false}
        />
        <MiniMap
          style={minimapStyle}
          nodeColor={nodeColor}
          maskColor="rgba(0, 0, 0, 0.1)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  )
}

export default GraphCanvas
