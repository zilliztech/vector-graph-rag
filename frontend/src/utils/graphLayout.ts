import dagre from '@dagrejs/dagre'
import type { Node, Edge } from '@xyflow/react'

const NODE_WIDTH = 120
const NODE_HEIGHT = 40

export interface LayoutOptions {
  direction?: 'TB' | 'BT' | 'LR' | 'RL'
  nodeSpacing?: number
  rankSpacing?: number
}

export function layoutGraph<T extends Record<string, unknown>, E extends Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge<E>[],
  options: LayoutOptions = {}
): Node<T>[] {
  const { direction = 'LR', nodeSpacing = 50, rankSpacing = 100 } = options

  if (nodes.length === 0) return []

  const g = new dagre.graphlib.Graph()
  g.setGraph({
    rankdir: direction,
    nodesep: nodeSpacing,
    ranksep: rankSpacing,
    marginx: 20,
    marginy: 20,
  })
  g.setDefaultEdgeLabel(() => ({}))

  // Add nodes
  nodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  })

  // Add edges
  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  // Run layout
  dagre.layout(g)

  // Apply positions to nodes
  return nodes.map((node) => {
    const nodeWithPosition = g.node(node.id)
    if (!nodeWithPosition) return node

    return {
      ...node,
      position: {
        x: nodeWithPosition.x - NODE_WIDTH / 2,
        y: nodeWithPosition.y - NODE_HEIGHT / 2,
      },
    }
  })
}

// Force-directed layout for circular arrangements
export function circularLayout<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  centerX: number = 300,
  centerY: number = 300,
  radius: number = 200
): Node<T>[] {
  const count = nodes.length
  if (count === 0) return []
  if (count === 1) {
    return [{ ...nodes[0], position: { x: centerX, y: centerY } }]
  }

  const angleStep = (2 * Math.PI) / count

  return nodes.map((node, index) => {
    const angle = index * angleStep - Math.PI / 2 // Start from top
    return {
      ...node,
      position: {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      },
    }
  })
}

// Grid layout
export function gridLayout<T extends Record<string, unknown>>(
  nodes: Node<T>[],
  columns: number = 4,
  cellWidth: number = 150,
  cellHeight: number = 80,
  startX: number = 50,
  startY: number = 50
): Node<T>[] {
  return nodes.map((node, index) => {
    const row = Math.floor(index / columns)
    const col = index % columns

    return {
      ...node,
      position: {
        x: startX + col * cellWidth,
        y: startY + row * cellHeight,
      },
    }
  })
}
